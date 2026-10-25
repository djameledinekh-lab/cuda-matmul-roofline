// FP32 square matrix multiplication C = A x B, three hand-written CUDA kernels:
//   1. naive      : one thread per output element, every operand read from global memory
//   2. tiled      : 32x32 tiles staged in shared memory, reused TILE_SIZE times
//   3. coarsened  : 64x64 data tiles, each thread computes a COARSE x COARSE block of outputs
// Exposed to Python through a PyTorch C++ extension (see bench.py).

#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32

#ifndef COARSE
#define COARSE 2
#endif
#define TILE_C 64                    // data tile of the coarsened kernel
#define BLOCK_C (TILE_C / COARSE)    // threads per block dimension

static_assert(TILE_C % COARSE == 0, "TILE_C must be divisible by COARSE");
static_assert(BLOCK_C <= 32, "BLOCK_C x BLOCK_C must stay <= 1024 threads per block");

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void matmul_coarsened_kernel(const float* A, const float* B, float* C, int N) {
    __shared__ float As[TILE_C][TILE_C];
    __shared__ float Bs[TILE_C][TILE_C];

    int blockRow = blockIdx.y * TILE_C;
    int blockCol = blockIdx.x * TILE_C;

    float acc[COARSE][COARSE];
    #pragma unroll
    for (int i = 0; i < COARSE; i++)
        #pragma unroll
        for (int j = 0; j < COARSE; j++)
            acc[i][j] = 0.0f;

    int numTiles = (N + TILE_C - 1) / TILE_C;
    for (int t = 0; t < numTiles; t++) {
        // Each thread loads a COARSE x COARSE patch of both shared tiles.
        #pragma unroll
        for (int i = 0; i < COARSE; i++) {
            #pragma unroll
            for (int j = 0; j < COARSE; j++) {
                int localRow = threadIdx.y * COARSE + i;
                int localCol = threadIdx.x * COARSE + j;

                int aRow = blockRow + localRow;
                int aCol = t * TILE_C + localCol;
                As[localRow][localCol] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0f;

                int bRow = t * TILE_C + localRow;
                int bCol = blockCol + localCol;
                Bs[localRow][localCol] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_C; k++) {
            float aReg[COARSE], bReg[COARSE];
            #pragma unroll
            for (int i = 0; i < COARSE; i++) aReg[i] = As[threadIdx.y * COARSE + i][k];
            #pragma unroll
            for (int j = 0; j < COARSE; j++) bReg[j] = Bs[k][threadIdx.x * COARSE + j];

            #pragma unroll
            for (int i = 0; i < COARSE; i++)
                #pragma unroll
                for (int j = 0; j < COARSE; j++)
                    acc[i][j] += aReg[i] * bReg[j];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < COARSE; i++) {
        #pragma unroll
        for (int j = 0; j < COARSE; j++) {
            int row = blockRow + threadIdx.y * COARSE + i;
            int col = blockCol + threadIdx.x * COARSE + j;
            if (row < N && col < N) C[row * N + col] = acc[i][j];
        }
    }
}

static void check_inputs(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.scalar_type() == torch::kFloat32 && B.scalar_type() == torch::kFloat32, "A and B must be float32");
    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1) && A.sizes() == B.sizes(), "A and B must be square and of the same size");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "A and B must be contiguous");
}

// Every kernel writes every in-bounds output element, so C can start uninitialized.
torch::Tensor matmul_naive(torch::Tensor A, torch::Tensor B) {
    check_inputs(A, B);
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);
    matmul_naive_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

torch::Tensor matmul_tiled(torch::Tensor A, torch::Tensor B) {
    check_inputs(A, B);
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

torch::Tensor matmul_coarsened(torch::Tensor A, torch::Tensor B) {
    check_inputs(A, B);
    int N = A.size(0);
    auto C = torch::empty({N, N}, A.options());
    dim3 threads(BLOCK_C, BLOCK_C);
    dim3 blocks((N + TILE_C - 1) / TILE_C, (N + TILE_C - 1) / TILE_C);
    matmul_coarsened_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_naive", &matmul_naive, "Naive FP32 matmul");
    m.def("matmul_tiled", &matmul_tiled, "Shared-memory tiled FP32 matmul (TILE_SIZE=32)");
    m.def("matmul_coarsened", &matmul_coarsened, "Tiled + thread-coarsened FP32 matmul (TILE_C=64)");
    m.attr("COARSE") = COARSE;
}
