//*****************************************************************************
// Copyright 2018-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <memory>
#include <vector>

#include "he_plaintext.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/element_type.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {
inline void avg_pool_seal(
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& arg_shape, const Shape& out_shape, const Shape& window_shape,
    const Strides& window_movement_strides, const Shape& padding_below,
    const Shape& padding_above, bool include_padding_in_avg_computation,
    const HESealBackend& he_seal_backend) {
  // At the outermost level we will walk over every output coordinate O.
  CoordinateTransform output_transform(out_shape);

  for (const Coordinate& out_coord : output_transform) {
    // Our output coordinate O will have the form:
    //
    //   (N,chan,i_1,...,i_n)

    size_t batch_index = out_coord[0];
    size_t channel = out_coord[1];

    // For the input data we need to iterate the coordinate:
    //
    //   I:
    //
    // over the range (noninclusive on the right):
    //
    //   (N,chan,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
    //
    //     (N+1,chan+1,s_1*i_1 + window_shape_1,...,s_n*i_n + window_shape_n)
    //
    // with unit stride.
    //
    // We iterate this over the *padded* data, so below we will need to check
    // for coordinates that fall in the padding area.

    size_t n_spatial_dimensions = arg_shape.size() - 2;

    Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
    Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
    Strides input_batch_transform_source_strides(2 + n_spatial_dimensions, 1);
    AxisVector input_batch_transform_source_axis_order(2 +
                                                       n_spatial_dimensions);
    CoordinateDiff input_batch_transform_padding_below(2 +
                                                       n_spatial_dimensions);
    CoordinateDiff input_batch_transform_padding_above(2 +
                                                       n_spatial_dimensions);

    input_batch_transform_start[0] = batch_index;
    input_batch_transform_end[0] = batch_index + 1;
    input_batch_transform_start[1] = channel;
    input_batch_transform_end[1] = channel + 1;
    input_batch_transform_padding_below[0] = 0;
    input_batch_transform_padding_below[1] = 0;
    input_batch_transform_padding_above[0] = 0;
    input_batch_transform_padding_above[1] = 0;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      size_t window_shape_this_dim = window_shape[i - 2];
      size_t movement_stride = window_movement_strides[i - 2];

      input_batch_transform_start[i] = movement_stride * out_coord[i];
      input_batch_transform_end[i] =
          input_batch_transform_start[i] + window_shape_this_dim;
      input_batch_transform_padding_below[i] = padding_below[i - 2];
      input_batch_transform_padding_above[i] = padding_above[i - 2];
    }

    for (size_t i = 0; i < arg_shape.size(); i++) {
      input_batch_transform_source_axis_order[i] = i;
    }

    CoordinateTransform input_batch_transform(
        arg_shape, input_batch_transform_start, input_batch_transform_end,
        input_batch_transform_source_strides,
        input_batch_transform_source_axis_order,
        input_batch_transform_padding_below,
        input_batch_transform_padding_above);

    // As we go, we compute the sum value:
    //
    //   output[O] := output[O] + arg[I]
    //
    // and the number of elements:
    //
    //   n_elements := n_elements + 1

    // T result = 0;
    std::shared_ptr<SealCiphertextWrapper> sum;
    bool first_add = true;

    size_t n_elements = 0;

    for (const Coordinate& input_batch_coord : input_batch_transform) {
      bool in_bounds =
          input_batch_transform.has_source_coordinate(input_batch_coord);

      if (in_bounds || include_padding_in_avg_computation) {
        // T v = in_bounds ?: 0;
        // result += v;

        if (first_add) {
          sum = arg[input_batch_transform.index(input_batch_coord)];
          first_add = false;
        } else {
          ngraph::he::scalar_add_seal(
              *sum, *arg[input_batch_transform.index(input_batch_coord)], sum,
              element::f32, he_seal_backend);
        }
        n_elements++;
      }
    }

    if (n_elements == 0) {
      throw std::runtime_error("AvgPool elements == 0, must be non-zero");
    }
    auto inv_n_elements = HEPlaintext({1.f / n_elements});

    ngraph::he::scalar_multiply_seal(*sum, inv_n_elements, sum, element::f32,
                                     he_seal_backend);

    out[output_transform.index(out_coord)] = sum;
  }
};

inline void avg_pool_seal(std::vector<HEPlaintext>& arg,
                          std::vector<HEPlaintext>& out, const Shape& arg_shape,
                          const Shape& out_shape, const Shape& window_shape,
                          const Strides& window_movement_strides,
                          const Shape& padding_below,
                          const Shape& padding_above,
                          bool include_padding_in_avg_computation,
                          const HESealBackend& he_seal_backend) {
  // At the outermost level we will walk over every output coordinate O.
  CoordinateTransform output_transform(out_shape);

  for (const Coordinate& out_coord : output_transform) {
    size_t batch_index = out_coord[0];
    size_t channel = out_coord[1];
    size_t n_spatial_dimensions = arg_shape.size() - 2;

    Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
    Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
    Strides input_batch_transform_source_strides(2 + n_spatial_dimensions, 1);
    AxisVector input_batch_transform_source_axis_order(2 +
                                                       n_spatial_dimensions);
    CoordinateDiff input_batch_transform_padding_below(2 +
                                                       n_spatial_dimensions);
    CoordinateDiff input_batch_transform_padding_above(2 +
                                                       n_spatial_dimensions);

    input_batch_transform_start[0] = batch_index;
    input_batch_transform_end[0] = batch_index + 1;
    input_batch_transform_start[1] = channel;
    input_batch_transform_end[1] = channel + 1;
    input_batch_transform_padding_below[0] = 0;
    input_batch_transform_padding_below[1] = 0;
    input_batch_transform_padding_above[0] = 0;
    input_batch_transform_padding_above[1] = 0;

    for (size_t i = 2; i < n_spatial_dimensions + 2; i++) {
      size_t window_shape_this_dim = window_shape[i - 2];
      size_t movement_stride = window_movement_strides[i - 2];

      input_batch_transform_start[i] = movement_stride * out_coord[i];
      input_batch_transform_end[i] =
          input_batch_transform_start[i] + window_shape_this_dim;
      input_batch_transform_padding_below[i] = padding_below[i - 2];
      input_batch_transform_padding_above[i] = padding_above[i - 2];
    }

    for (size_t i = 0; i < arg_shape.size(); i++) {
      input_batch_transform_source_axis_order[i] = i;
    }

    CoordinateTransform input_batch_transform(
        arg_shape, input_batch_transform_start, input_batch_transform_end,
        input_batch_transform_source_strides,
        input_batch_transform_source_axis_order,
        input_batch_transform_padding_below,
        input_batch_transform_padding_above);

    // T result = 0;
    HEPlaintext sum;
    bool first_add = true;

    size_t n_elements = 0;

    for (const Coordinate& input_batch_coord : input_batch_transform) {
      bool in_bounds =
          input_batch_transform.has_source_coordinate(input_batch_coord);

      if (in_bounds || include_padding_in_avg_computation) {
        // T v = in_bounds ?: 0;
        // result += v;
        if (first_add) {
          sum = arg[input_batch_transform.index(input_batch_coord)];
          first_add = false;
        } else {
          ngraph::he::scalar_add_seal(
              sum, arg[input_batch_transform.index(input_batch_coord)], sum,
              element::f32, he_seal_backend);
        }
        n_elements++;
      }
    }

    if (n_elements == 0) {
      throw std::runtime_error("AvgPool elements == 0, must be non-zero");
    }
    auto inv_n_elements = HEPlaintext({1.f / n_elements});

    ngraph::he::scalar_multiply_seal(sum, inv_n_elements, sum, element::f32,
                                     he_seal_backend);

    out[output_transform.index(out_coord)] = sum;
  }
};
}  // namespace he
}  // namespace ngraph
