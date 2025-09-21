#pragma once
#include "cpu/vision.h"
// 高版本torch弃用了THC
// #include "ATen/cuda/DeviceUtils.cuh"
#include <c10/cuda/CUDACachingAllocator.h>

#ifdef WITH_CUDA
#include "cuda/vision.h"
// #include <THC/THC.h>
// extern THCState *state;
#endif

/*
适配参考资料：
  - https://blog.csdn.net/weixin_41868417/article/details/123819183
*/



int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{

    // TODO check dimensions
    long batch, ref_nb, query_nb, dim, k;
    batch = ref.size(0);
    dim = ref.size(1);
    k = idx.size(1);
    ref_nb = ref.size(2);
    query_nb = query.size(2);

    float *ref_dev = ref.data<float>();
    float *query_dev = query.data<float>();
    long *idx_dev = idx.data<long>();




  // if (ref.type().is_cuda()) {
  if (ref.is_cuda()) {
#ifdef WITH_CUDA
    // TODO raise error if not compiled with CUDA
    // adapt for cuda torch 2.0.0 高版本弃用了THC
    // float *dist_dev = (float*)THCudaMalloc(state, ref_nb * query_nb * sizeof(float));
    float *dist_dev = (float*)c10::cuda::CUDACachingAllocator::raw_alloc(ref_nb * query_nb * sizeof(float));

    for (int b = 0; b < batch; b++)
    {
    // knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
    //   dist_dev, idx_dev + b * k * query_nb, THCState_getCurrentStream(state));
      knn_device(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, c10::cuda::getCurrentCUDAStream());
    }
    // THCudaFree(state, dist_dev);
    // 高版本torch弃用了THC
    c10::cuda::CUDACachingAllocator::raw_delete(dist_dev);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess)
    // {
    //     printf("error in knn: %s\n", cudaGetErrorString(err));
    //     // THError("aborting");
    // }
    return 1;
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }


    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    long *ind_buf = (long*)malloc(ref_nb * sizeof(long));
    for (int b = 0; b < batch; b++) {
    knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, query_dev + b * dim * query_nb, query_nb, dim, k,
      dist_dev, idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;

}
