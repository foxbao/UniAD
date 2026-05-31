//
// Created by Derry Lin on 2022/11/21.
//

#ifndef TENSORRT_OPS_ROTATEKERNEL_H
#define TENSORRT_OPS_ROTATEKERNEL_H

#include "cuda_int8.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <type_traits>

typedef std::conditional<NV_TENSORRT_MAJOR<10, int, int64_t>::type TRT_INT_TYPE;

enum class RotateInterpolation { Bilinear, Nearest };

template <typename T>
void rotate(T *output, T *input, T *angle, T *center, TRT_INT_TYPE *input_dims,
            RotateInterpolation interp, cudaStream_t stream);

void rotate_h2(__half2 *output, __half2 *input, __half *angle, __half *center,
               TRT_INT_TYPE *input_dims, RotateInterpolation interp,
               cudaStream_t stream);

template <typename T>
void rotate_int8(int8_4 *output, float scale_o, const int8_4 *input,
                 float scale_i, const T *angle, const T *center,
                 TRT_INT_TYPE *input_dims, RotateInterpolation interp,
                 cudaStream_t stream);

#endif // TENSORRT_OPS_ROTATEKERNEL_H
