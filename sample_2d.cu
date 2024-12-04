#include <stdio.h>

#define N 8

__global__ void matrixFill(int *x, int *y, double *a) {
  int col = (blockIdx.x * blockDim.x) + threadIdx.x;
  int row = (blockIdx.y * blockDim.y) + threadIdx.y;

  int idx = col * N + row;
  if (col < N && row < N) {
    // x[idx] = blockIdx.x;
    // y[idx] = blockIdx.y;
    // x[idx] = blockDim.x;
    // y[idx] = blockDim.y;
    x[idx] = threadIdx.x;
    y[idx] = threadIdx.y;

    a[idx] = row * 0.01 + col;
  }
}

int main() {
  int *x, *y;
  double *a;
  int *dev_x, *dev_y;
  double *dev_a;

  x = (int *)malloc(N * N * sizeof(int));
  y = (int *)malloc(N * N * sizeof(int));
  a = (double *)malloc(N * N * sizeof(double));
  cudaMalloc(&dev_x, N * N * sizeof(int));
  cudaMalloc(&dev_y, N * N * sizeof(int));
  cudaMalloc(&dev_a, N * N * sizeof(double));
  dim3 dimBlock(2, 2);
  dim3 dimThread(4, 4);
  matrixFill<<<dimBlock, dimThread>>>(dev_x, dev_y, dev_a);
  cudaMemcpy(x, dev_x, N * N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, dev_y, N * N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(a, dev_a, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; ++j) {
      int idx = j + i * N;
      printf("(%d,%d)", x[idx], y[idx]);
    }
    printf("\n");
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; ++j) {
      int idx = j + i * N;
      printf("%f\t", a[idx]);
    }
    printf("\n");
  }
  cudaFree(dev_x);
  cudaFree(dev_y);
  cudaFree(dev_a);
  free(x);
  free(y);
  free(a);
}