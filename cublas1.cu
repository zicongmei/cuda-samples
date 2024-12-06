#include <cstdio>
#include <cstdlib>
#include <vector>
#include <stdexcept>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


using data_type = double;

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    /*
     *   A = | 1.0 2.0 3.0 4.0 |
     */

    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    const int incx = 1;

    data_type result = 0.0;

    data_type *d_A = nullptr;

    printf("A\n");
    for (size_t i = 0; i< A.size(); ++i){
        printf("%f, ",A[i]);
    }
    printf("\n");
    //print_vector(A.size(), A.data());
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(data_type) * A.size()));

    CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(data_type) * A.size(), cudaMemcpyHostToDevice,
                               stream));

    /* step 3: compute */
    CUBLAS_CHECK(cublasDasum(cublasH, A.size(), d_A, incx, &result));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    /*
     *   result = 10.00
     */

    printf("result\n");
    std::printf("%0.2f\n", result);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}