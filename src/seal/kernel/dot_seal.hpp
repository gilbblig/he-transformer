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
#include "seal/he_seal_backend.hpp"
#include "seal/kernel/add_seal.hpp"
#include "seal/kernel/multiply_seal.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"

namespace ngraph {
namespace he {
// TODO: templatize?
inline void dot_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& arg0_shape, const Shape& arg1_shape, const Shape& out_shape,
    size_t reduction_axes_count, const element::Type& element_type,
    const HESealBackend& he_seal_backend) {
  // Get the sizes of the dot axes. It's easiest to pull them from arg1
  // because they're right up front.
  Shape dot_axis_sizes(reduction_axes_count);
  std::copy(arg1_shape.begin(), arg1_shape.begin() + reduction_axes_count,
            dot_axis_sizes.begin());

  CoordinateTransform arg0_transform(arg0_shape);
  CoordinateTransform arg1_transform(arg1_shape);
  CoordinateTransform output_transform(out_shape);

  // Create coordinate transforms for arg0 and arg1 that throw away the dotted
  // axes.
  size_t arg0_projected_rank = arg0_shape.size() - reduction_axes_count;
  size_t arg1_projected_rank = arg1_shape.size() - reduction_axes_count;

  Shape arg0_projected_shape(arg0_projected_rank);
  std::copy(arg0_shape.begin(), arg0_shape.begin() + arg0_projected_rank,
            arg0_projected_shape.begin());

  Shape arg1_projected_shape(arg1_projected_rank);
  std::copy(arg1_shape.begin() + reduction_axes_count, arg1_shape.end(),
            arg1_projected_shape.begin());

  CoordinateTransform arg0_projected_transform(arg0_projected_shape);
  CoordinateTransform arg1_projected_transform(arg1_projected_shape);

  // Create a coordinate transform that allows us to iterate over all possible
  // values
  // for the dotted axes.
  CoordinateTransform dot_axes_transform(dot_axis_sizes);

  // Get arg0_projected_size and arg1_projected_size for parallelization
  // and pre-compute coordinates
  std::vector<ngraph::Coordinate> arg0_projected_coords;
  for (const Coordinate& coord : arg0_projected_transform) {
    arg0_projected_coords.emplace_back(coord);
  }

  std::vector<ngraph::Coordinate> arg1_projected_coords;
  for (const Coordinate& coord : arg1_projected_transform) {
    arg1_projected_coords.emplace_back(coord);
  }

  size_t arg0_projected_size = arg0_projected_coords.size();
  size_t arg1_projected_size = arg1_projected_coords.size();
  size_t global_projected_size = arg0_projected_size * arg1_projected_size;

  // TODO: don't create new thread for every loop index, only one per thread
#pragma omp parallel for
  for (size_t global_projected_idx = 0;
       global_projected_idx < global_projected_size; ++global_projected_idx) {
    // Init thread-local memory pool for each thread
    seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();

    // Compute outer and inner index
    size_t arg0_projected_idx = global_projected_idx / arg1_projected_size;
    size_t arg1_projected_idx = global_projected_idx % arg1_projected_size;

    const Coordinate& arg0_projected_coord =
        arg0_projected_coords[arg0_projected_idx];
    const Coordinate& arg1_projected_coord =
        arg1_projected_coords[arg1_projected_idx];

    // The output coordinate is just the concatenation of the projected
    // coordinates.
    Coordinate out_coord(arg0_projected_coord.size() +
                         arg1_projected_coord.size());

    auto out_coord_it =
        std::copy(arg0_projected_coord.begin(), arg0_projected_coord.end(),
                  out_coord.begin());
    std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
              out_coord_it);

    size_t out_index = output_transform.index(out_coord);

    // Walk along the dotted axes.
    Coordinate arg0_coord(arg0_shape.size());
    Coordinate arg1_coord(arg1_shape.size());
    auto arg0_it = std::copy(arg0_projected_coord.begin(),
                             arg0_projected_coord.end(), arg0_coord.begin());

    std::shared_ptr<SealCiphertextWrapper> sum;
    bool first_add = true;

    for (const Coordinate& dot_axis_positions : dot_axes_transform) {
      // In order to find the points to multiply together, we need to inject
      // our current positions along the dotted axes back into the projected
      // arg0 and arg1 coordinates.
      std::copy(dot_axis_positions.begin(), dot_axis_positions.end(), arg0_it);

      auto arg1_it = std::copy(dot_axis_positions.begin(),
                               dot_axis_positions.end(), arg1_coord.begin());
      std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
                arg1_it);

      // Multiply and add to the summands.
      auto mult_arg0 = *arg0[arg0_transform.index(arg0_coord)];
      auto mult_arg1 = *arg1[arg1_transform.index(arg1_coord)];
      auto prod = he_seal_backend.create_empty_ciphertext();
      scalar_multiply_seal(mult_arg0, mult_arg1, prod, element_type,
                           he_seal_backend, pool);
      if (first_add) {
        sum = prod;
        first_add = false;
      } else {
        scalar_add_seal(*prod, *sum, sum, element_type, he_seal_backend, pool);
      }
    }
    // Write the sum back.
    if (first_add) {
      out[out_index] = std::make_shared<SealCiphertextWrapper>();
      out[out_index]->known_value() = true;
      out[out_index]->value() = 0;
    } else {
      out[out_index] = sum;
    }
  }
}
// End CCC

// PCC
inline void dot_seal(
    const std::vector<HEPlaintext>& arg0,
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& arg0_shape, const Shape& arg1_shape, const Shape& out_shape,
    size_t reduction_axes_count, const element::Type& element_type,
    const HESealBackend& he_seal_backend) {
  // Get the sizes of the dot axes. It's easiest to pull them from arg1
  // because they're right up front.
  Shape dot_axis_sizes(reduction_axes_count);
  std::copy(arg1_shape.begin(), arg1_shape.begin() + reduction_axes_count,
            dot_axis_sizes.begin());

  CoordinateTransform arg0_transform(arg0_shape);
  CoordinateTransform arg1_transform(arg1_shape);
  CoordinateTransform output_transform(out_shape);

  // Create coordinate transforms for arg0 and arg1 that throw away the dotted
  // axes.
  size_t arg0_projected_rank = arg0_shape.size() - reduction_axes_count;
  size_t arg1_projected_rank = arg1_shape.size() - reduction_axes_count;

  Shape arg0_projected_shape(arg0_projected_rank);
  std::copy(arg0_shape.begin(), arg0_shape.begin() + arg0_projected_rank,
            arg0_projected_shape.begin());

  Shape arg1_projected_shape(arg1_projected_rank);
  std::copy(arg1_shape.begin() + reduction_axes_count, arg1_shape.end(),
            arg1_projected_shape.begin());

  CoordinateTransform arg0_projected_transform(arg0_projected_shape);
  CoordinateTransform arg1_projected_transform(arg1_projected_shape);

  // Create a coordinate transform that allows us to iterate over all possible
  // values
  // for the dotted axes.
  CoordinateTransform dot_axes_transform(dot_axis_sizes);

  // Get arg0_projected_size and arg1_projected_size for parallelization
  // and pre-compute coordinates
  std::vector<ngraph::Coordinate> arg0_projected_coords;
  for (const Coordinate& coord : arg0_projected_transform) {
    arg0_projected_coords.emplace_back(coord);
  }

  std::vector<ngraph::Coordinate> arg1_projected_coords;
  for (const Coordinate& coord : arg1_projected_transform) {
    arg1_projected_coords.emplace_back(coord);
  }

  size_t arg0_projected_size = arg0_projected_coords.size();
  size_t arg1_projected_size = arg1_projected_coords.size();
  size_t global_projected_size = arg0_projected_size * arg1_projected_size;

// TODO: don't create new thread for every loop index, only one per thread
#pragma omp parallel for
  for (size_t global_projected_idx = 0;
       global_projected_idx < global_projected_size; ++global_projected_idx) {
    // Init thread-local memory pool for each thread
    seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();

    // Compute outer and inner index
    size_t arg0_projected_idx = global_projected_idx / arg1_projected_size;
    size_t arg1_projected_idx = global_projected_idx % arg1_projected_size;

    const Coordinate& arg0_projected_coord =
        arg0_projected_coords[arg0_projected_idx];
    const Coordinate& arg1_projected_coord =
        arg1_projected_coords[arg1_projected_idx];

    // The output coordinate is just the concatenation of the projected
    // coordinates.
    Coordinate out_coord(arg0_projected_coord.size() +
                         arg1_projected_coord.size());

    auto out_coord_it =
        std::copy(arg0_projected_coord.begin(), arg0_projected_coord.end(),
                  out_coord.begin());
    std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
              out_coord_it);

    size_t out_index = output_transform.index(out_coord);

    // Walk along the dotted axes.
    Coordinate arg0_coord(arg0_shape.size());
    Coordinate arg1_coord(arg1_shape.size());
    auto arg0_it = std::copy(arg0_projected_coord.begin(),
                             arg0_projected_coord.end(), arg0_coord.begin());

    std::shared_ptr<SealCiphertextWrapper> sum;
    bool first_add = true;

    for (const Coordinate& dot_axis_positions : dot_axes_transform) {
      // In order to find the points to multiply together, we need to inject
      // our current positions along the dotted axes back into the projected
      // arg0 and arg1 coordinates.
      std::copy(dot_axis_positions.begin(), dot_axis_positions.end(), arg0_it);

      auto arg1_it = std::copy(dot_axis_positions.begin(),
                               dot_axis_positions.end(), arg1_coord.begin());
      std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
                arg1_it);

      // Multiply and add to the summands.
      auto mult_arg0 = arg0[arg0_transform.index(arg0_coord)];
      auto mult_arg1 = arg1[arg1_transform.index(arg1_coord)];
      auto prod = he_seal_backend.create_empty_ciphertext();
      scalar_multiply_seal(mult_arg0, *mult_arg1, prod, element_type,
                           he_seal_backend, pool);
      if (first_add) {
        // TODO: std::move(prod)?
        sum = prod;
        first_add = false;
      } else {
        scalar_add_seal(*sum, *prod, sum, element_type, he_seal_backend, pool);
      }
    }
    // Write the sum back.
    if (first_add) {
      out[out_index] = std::make_shared<SealCiphertextWrapper>();
      out[out_index]->known_value() = true;
      out[out_index]->value() = 0;
    } else {
      out[out_index] = sum;
    }
  }
}

inline void dot_seal(
    const std::vector<std::shared_ptr<SealCiphertextWrapper>>& arg0,
    const std::vector<HEPlaintext>& arg1,
    std::vector<std::shared_ptr<SealCiphertextWrapper>>& out,
    const Shape& arg0_shape, const Shape& arg1_shape, const Shape& out_shape,
    size_t reduction_axes_count, const element::Type& element_type,
    const HESealBackend& he_seal_backend) {
  // Get the sizes of the dot axes. It's easiest to pull them from arg1
  // because they're right up front.
  Shape dot_axis_sizes(reduction_axes_count);
  std::copy(arg1_shape.begin(), arg1_shape.begin() + reduction_axes_count,
            dot_axis_sizes.begin());

  CoordinateTransform arg0_transform(arg0_shape);
  CoordinateTransform arg1_transform(arg1_shape);
  CoordinateTransform output_transform(out_shape);

  // Create coordinate transforms for arg0 and arg1 that throw away the dotted
  // axes.
  size_t arg0_projected_rank = arg0_shape.size() - reduction_axes_count;
  size_t arg1_projected_rank = arg1_shape.size() - reduction_axes_count;

  Shape arg0_projected_shape(arg0_projected_rank);
  std::copy(arg0_shape.begin(), arg0_shape.begin() + arg0_projected_rank,
            arg0_projected_shape.begin());

  Shape arg1_projected_shape(arg1_projected_rank);
  std::copy(arg1_shape.begin() + reduction_axes_count, arg1_shape.end(),
            arg1_projected_shape.begin());

  CoordinateTransform arg0_projected_transform(arg0_projected_shape);
  CoordinateTransform arg1_projected_transform(arg1_projected_shape);

  // Create a coordinate transform that allows us to iterate over all possible
  // values
  // for the dotted axes.
  CoordinateTransform dot_axes_transform(dot_axis_sizes);

  // Get arg0_projected_size and arg1_projected_size for parallelization
  // and pre-compute coordinates
  std::vector<ngraph::Coordinate> arg0_projected_coords;
  for (const Coordinate& coord : arg0_projected_transform) {
    arg0_projected_coords.emplace_back(coord);
  }

  std::vector<ngraph::Coordinate> arg1_projected_coords;
  for (const Coordinate& coord : arg1_projected_transform) {
    arg1_projected_coords.emplace_back(coord);
  }

  size_t arg0_projected_size = arg0_projected_coords.size();
  size_t arg1_projected_size = arg1_projected_coords.size();
  size_t global_projected_size = arg0_projected_size * arg1_projected_size;

// TODO: don't create new thread for every loop index, only one per thread
#pragma omp parallel for
  for (size_t global_projected_idx = 0;
       global_projected_idx < global_projected_size; ++global_projected_idx) {
    // Init thread-local memory pool for each thread
    seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();

    // Compute outer and inner index
    size_t arg0_projected_idx = global_projected_idx / arg1_projected_size;
    size_t arg1_projected_idx = global_projected_idx % arg1_projected_size;

    const Coordinate& arg0_projected_coord =
        arg0_projected_coords[arg0_projected_idx];
    const Coordinate& arg1_projected_coord =
        arg1_projected_coords[arg1_projected_idx];

    // The output coordinate is just the concatenation of the projected
    // coordinates.
    Coordinate out_coord(arg0_projected_coord.size() +
                         arg1_projected_coord.size());

    auto out_coord_it =
        std::copy(arg0_projected_coord.begin(), arg0_projected_coord.end(),
                  out_coord.begin());
    std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
              out_coord_it);

    size_t out_index = output_transform.index(out_coord);

    // Walk along the dotted axes.
    Coordinate arg0_coord(arg0_shape.size());
    Coordinate arg1_coord(arg1_shape.size());
    auto arg0_it = std::copy(arg0_projected_coord.begin(),
                             arg0_projected_coord.end(), arg0_coord.begin());

    std::shared_ptr<SealCiphertextWrapper> sum;
    bool first_add = true;

    for (const Coordinate& dot_axis_positions : dot_axes_transform) {
      // In order to find the points to multiply together, we need to inject
      // our current positions along the dotted axes back into the projected
      // arg0 and arg1 coordinates.
      std::copy(dot_axis_positions.begin(), dot_axis_positions.end(), arg0_it);

      auto arg1_it = std::copy(dot_axis_positions.begin(),
                               dot_axis_positions.end(), arg1_coord.begin());
      std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
                arg1_it);

      // Multiply and add to the summands.
      auto mult_arg0 = arg0[arg0_transform.index(arg0_coord)];
      auto mult_arg1 = arg1[arg1_transform.index(arg1_coord)];
      auto prod = he_seal_backend.create_empty_ciphertext();
      scalar_multiply_seal(*mult_arg0, mult_arg1, prod, element_type,
                           he_seal_backend, pool);
      if (first_add) {
        // TODO: std::move(prod)?
        sum = prod;
        first_add = false;
      } else {
        scalar_add_seal(*prod, *sum, sum, element_type, he_seal_backend, pool);
      }
    }
    // Write the sum back.
    if (first_add) {
      out[out_index] = std::make_shared<SealCiphertextWrapper>();
      out[out_index]->known_value() = true;
      out[out_index]->value() = 0;
    } else {
      out[out_index] = sum;
    }
  }
}

// PPP
inline void dot_seal(const std::vector<HEPlaintext>& arg0,
                     const std::vector<HEPlaintext>& arg1,
                     std::vector<HEPlaintext>& out, const Shape& arg0_shape,
                     const Shape& arg1_shape, const Shape& out_shape,
                     size_t reduction_axes_count,
                     const element::Type& element_type,
                     const HESealBackend& he_seal_backend) {
  // Get the sizes of the dot axes. It's easiest to pull them from arg1
  // because they're right up front.
  Shape dot_axis_sizes(reduction_axes_count);
  std::copy(arg1_shape.begin(), arg1_shape.begin() + reduction_axes_count,
            dot_axis_sizes.begin());

  CoordinateTransform arg0_transform(arg0_shape);
  CoordinateTransform arg1_transform(arg1_shape);
  CoordinateTransform output_transform(out_shape);

  // Create coordinate transforms for arg0 and arg1 that throw away the dotted
  // axes.
  size_t arg0_projected_rank = arg0_shape.size() - reduction_axes_count;
  size_t arg1_projected_rank = arg1_shape.size() - reduction_axes_count;

  Shape arg0_projected_shape(arg0_projected_rank);
  std::copy(arg0_shape.begin(), arg0_shape.begin() + arg0_projected_rank,
            arg0_projected_shape.begin());

  Shape arg1_projected_shape(arg1_projected_rank);
  std::copy(arg1_shape.begin() + reduction_axes_count, arg1_shape.end(),
            arg1_projected_shape.begin());

  CoordinateTransform arg0_projected_transform(arg0_projected_shape);
  CoordinateTransform arg1_projected_transform(arg1_projected_shape);

  // Create a coordinate transform that allows us to iterate over all possible
  // values
  // for the dotted axes.
  CoordinateTransform dot_axes_transform(dot_axis_sizes);

  // Get arg0_projected_size and arg1_projected_size for parallelization
  // and pre-compute coordinates
  std::vector<ngraph::Coordinate> arg0_projected_coords;
  for (const Coordinate& coord : arg0_projected_transform) {
    arg0_projected_coords.emplace_back(coord);
  }

  std::vector<ngraph::Coordinate> arg1_projected_coords;
  for (const Coordinate& coord : arg1_projected_transform) {
    arg1_projected_coords.emplace_back(coord);
  }

  size_t arg0_projected_size = arg0_projected_coords.size();
  size_t arg1_projected_size = arg1_projected_coords.size();
  size_t global_projected_size = arg0_projected_size * arg1_projected_size;

// TODO: don't create new thread for every loop index, only one per thread
#pragma omp parallel for
  for (size_t global_projected_idx = 0;
       global_projected_idx < global_projected_size; ++global_projected_idx) {
    // Init thread-local memory pool for each thread
    seal::MemoryPoolHandle pool = seal::MemoryPoolHandle::ThreadLocal();

    // Compute outer and inner index
    size_t arg0_projected_idx = global_projected_idx / arg1_projected_size;
    size_t arg1_projected_idx = global_projected_idx % arg1_projected_size;

    const Coordinate& arg0_projected_coord =
        arg0_projected_coords[arg0_projected_idx];
    const Coordinate& arg1_projected_coord =
        arg1_projected_coords[arg1_projected_idx];

    // The output coordinate is just the concatenation of the projected
    // coordinates.
    Coordinate out_coord(arg0_projected_coord.size() +
                         arg1_projected_coord.size());

    auto out_coord_it =
        std::copy(arg0_projected_coord.begin(), arg0_projected_coord.end(),
                  out_coord.begin());
    std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
              out_coord_it);

    size_t out_index = output_transform.index(out_coord);

    // Walk along the dotted axes.
    Coordinate arg0_coord(arg0_shape.size());
    Coordinate arg1_coord(arg1_shape.size());
    auto arg0_it = std::copy(arg0_projected_coord.begin(),
                             arg0_projected_coord.end(), arg0_coord.begin());

    auto sum = HEPlaintext();
    bool first_add = true;

    for (const Coordinate& dot_axis_positions : dot_axes_transform) {
      // In order to find the points to multiply together, we need to inject
      // our current positions along the dotted axes back into the projected
      // arg0 and arg1 coordinates.
      std::copy(dot_axis_positions.begin(), dot_axis_positions.end(), arg0_it);

      auto arg1_it = std::copy(dot_axis_positions.begin(),
                               dot_axis_positions.end(), arg1_coord.begin());
      std::copy(arg1_projected_coord.begin(), arg1_projected_coord.end(),
                arg1_it);

      // Multiply and add to the summands.
      auto mult_arg0 = arg0[arg0_transform.index(arg0_coord)];
      auto mult_arg1 = arg1[arg1_transform.index(arg1_coord)];
      auto prod = HEPlaintext();
      scalar_multiply_seal(mult_arg0, mult_arg1, prod, element_type,
                           he_seal_backend);
      if (first_add) {
        sum = prod;
        first_add = false;
      } else {
        scalar_add_seal(prod, sum, sum, element_type, he_seal_backend);
      }
    }
    // Write the sum back.
    out[out_index] = sum;
  }
}
}  // namespace he
}  // namespace ngraph
