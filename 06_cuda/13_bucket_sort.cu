#include <cstdio>
#include <cstdlib>
#include <vector>
__global__ void init_bucket(int *bucket, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < range) {
    bucket[i] = 0;
  }
}

__global__ void count_bucket(int *key, int *bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    atomicAdd(&bucket[key[i]], 1);
  }
}

__global__ void fill_key(int *key, int *bucket, int *offset, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < range) {
    for (int j = 0; j < bucket[i]; j++) {
      key[offset[i] + j] = i;
    }
  }
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *d_key;
  int *d_bucket;
  int *d_offset;

  cudaMallocManaged(&d_key, n * sizeof(int));
  cudaMallocManaged(&d_bucket, range * sizeof(int));
  cudaMallocManaged(&d_offset, range * sizeof(int));

  // std::vector<int> key から Unified Memory へコピー
  for (int i = 0; i < n; i++) {
    d_key[i] = key[i];
  }

  int blockSize = 256;
  int gridN = (n + blockSize - 1) / blockSize;
  int gridRange = (range + blockSize - 1) / blockSize;

  // bucket初期化
  init_bucket<<<gridRange, blockSize>>>(d_bucket, range);
  cudaDeviceSynchronize();

  // bucket[key[i]]++ を並列化
  count_bucket<<<gridN, blockSize>>>(d_key, d_bucket, n);
  cudaDeviceSynchronize();

  // bucket から offset を作る
  d_offset[0] = 0;
  for (int i = 1; i < range; i++) {
    d_offset[i] = d_offset[i - 1] + d_bucket[i - 1];
  }

  // bucketの内容に従って key を並列に埋め直す
  fill_key<<<gridRange, blockSize>>>(d_key, d_bucket, d_offset, range);
  cudaDeviceSynchronize();

  // Unified Memory から std::vector<int> key に戻す
  for (int i = 0; i < n; i++) {
    key[i] = d_key[i];
  }

  cudaFree(d_key);
  cudaFree(d_bucket);
  cudaFree(d_offset);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
