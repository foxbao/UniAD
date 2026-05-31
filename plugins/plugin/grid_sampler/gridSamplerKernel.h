//
// Created by Derry Lin on 2022/10/22.
//

#ifndef TENSORRT_OPS_GRIDSAMPLERKERNEL_H
#define TENSORRT_OPS_GRIDSAMPLERKERNEL_H

#include "cuda_int8.h"
#include <cuda_runtime.h>
#include <type_traits>
#include <NvInfer.h>

typedef std::conditional<NV_TENSORRT_MAJOR<10, int, int64_t>::type TRT_INT_TYPE;

enum class GridSamplerInterpolation { Bilinear, Nearest, Bicubic };
enum class GridSamplerPadding { Zeros, Border, Reflection };

template <typename T>
void grid_sample(T *output, const T *input, const T *grid, TRT_INT_TYPE *output_dims,
                 TRT_INT_TYPE *input_dims, TRT_INT_TYPE *grid_dims, int nb_dims,
                 GridSamplerInterpolation interp, GridSamplerPadding padding,
                 bool align_corners, cudaStream_t stream);

void grid_sample_int8(int8_4 *output, const float &scale_o, const int8_4 *input,
                      const float &scale_i, const int8_4 *grid,
                      const float &scale_g, TRT_INT_TYPE *output_dims, TRT_INT_TYPE *input_dims,
                      TRT_INT_TYPE *grid_dims, int nb_dims,
                      GridSamplerInterpolation interp,
                      GridSamplerPadding padding, bool align_corners,
                      cudaStream_t stream);

#endif // TENSORRT_OPS_GRIDSAMPLERKERNEL_H
