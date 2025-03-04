//*****************************************************************************
// Copyright 2018 Intel Corporation
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <chrono>
#include <limits>
#include <utility>

#include "ngraph/runtime/tensor.hpp"
#include "seal/seal_util.hpp"
#include "seal/util/uintarith.h"

#include "seal/he_seal_backend.hpp"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "seal/util/polyarithsmallmod.h"

// Matches the modulus chain for the two elements in-place
// The elements are modified if necessary
void ngraph::he::match_modulus_and_scale_inplace(
    SealCiphertextWrapper& arg0, SealCiphertextWrapper& arg1,
    const HESealBackend& he_seal_backend, seal::MemoryPoolHandle pool) {
  size_t chain_ind0 = ngraph::he::get_chain_index(arg0, he_seal_backend);
  size_t chain_ind1 = ngraph::he::get_chain_index(arg1, he_seal_backend);

  if (chain_ind0 == chain_ind1) {
    return;
  }
  bool rescale = !ngraph::he::within_rescale_tolerance(arg0, arg1);

  if (chain_ind0 < chain_ind1) {
    auto arg0_parms_id = arg0.ciphertext().parms_id();
    if (rescale) {
      he_seal_backend.get_evaluator()->rescale_to_inplace(arg1.ciphertext(),
                                                          arg0_parms_id);
    } else {
      he_seal_backend.get_evaluator()->mod_switch_to_inplace(arg1.ciphertext(),
                                                             arg0_parms_id);
    }
    chain_ind1 = ngraph::he::get_chain_index(arg1, he_seal_backend);
  } else {  // chain_ind0 > chain_ind1
    auto arg1_parms_id = arg1.ciphertext().parms_id();
    if (rescale) {
      he_seal_backend.get_evaluator()->rescale_to_inplace(arg0.ciphertext(),
                                                          arg1_parms_id);
    } else {
      he_seal_backend.get_evaluator()->mod_switch_to_inplace(arg0.ciphertext(),
                                                             arg1_parms_id);
    }
    chain_ind0 = ngraph::he::get_chain_index(arg0, he_seal_backend);
  }
  NGRAPH_CHECK(chain_ind0 == chain_ind1);
  ngraph::he::match_scale(arg0, arg1, he_seal_backend);
}

// Encode value into vector of coefficients
void ngraph::he::encode(double value, double scale,
                        seal::parms_id_type parms_id,
                        std::vector<std::uint64_t>& destination,
                        const HESealBackend& he_seal_backend,
                        seal::MemoryPoolHandle pool) {
  // Verify parameters.
  auto context = he_seal_backend.get_context();
  auto context_data_ptr = context->get_context_data(parms_id);
  if (!context_data_ptr) {
    throw ngraph_error("parms_id is not valid for encryption parameters");
  }
  if (!pool) {
    throw ngraph_error("pool is uninitialized");
  }

  auto& context_data = *context_data_ptr;
  auto& parms = context_data.parms();
  auto& coeff_modulus = parms.coeff_modulus();
  size_t coeff_mod_count = coeff_modulus.size();
  size_t coeff_count = parms.poly_modulus_degree();

  // Quick sanity check
  if (!seal::util::product_fits_in(coeff_mod_count, coeff_count)) {
    throw ngraph_error("invalid parameters");
  }

  // Check that scale is positive and not too large
  if (scale <= 0 || (static_cast<int>(log2(scale)) >=
                     context_data.total_coeff_modulus_bit_count())) {
    NGRAPH_INFO << "scale " << scale;
    NGRAPH_INFO << "context_data.total_coeff_modulus_bit_count"
                << context_data.total_coeff_modulus_bit_count();
    throw ngraph_error("scale out of bounds");
  }

  // Compute the scaled value
  value *= scale;

  int coeff_bit_count = static_cast<int>(log2(fabs(value))) + 2;
  if (coeff_bit_count >= context_data.total_coeff_modulus_bit_count()) {
#pragma omp critical
    {
      NGRAPH_INFO << "Failed to encode " << value / scale << " at scale "
                  << scale;
      NGRAPH_INFO << "coeff_bit_count " << coeff_bit_count;
      NGRAPH_INFO << "coeff_mod_count " << coeff_mod_count;
      NGRAPH_INFO << "total coeff modulus bit count "
                  << context_data.total_coeff_modulus_bit_count();
      throw ngraph_error("encoded value is too large");
    }
  }

  double two_pow_64 = pow(2.0, 64);

  // Resize destination to appropriate size
  // TODO: use reserve?
  destination.resize(coeff_mod_count);

  double coeffd = round(value);
  bool is_negative = std::signbit(coeffd);
  coeffd = fabs(coeffd);

  // Use faster decomposition methods when possible
  if (coeff_bit_count <= 64) {
    uint64_t coeffu = static_cast<uint64_t>(fabs(coeffd));

    if (is_negative) {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] = seal::util::negate_uint_mod(
            coeffu % coeff_modulus[j].value(), coeff_modulus[j]);
      }
    } else {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] = coeffu % coeff_modulus[j].value();
      }
    }
  } else if (coeff_bit_count <= 128) {
    uint64_t coeffu[2]{static_cast<uint64_t>(fmod(coeffd, two_pow_64)),
                       static_cast<uint64_t>(coeffd / two_pow_64)};

    if (is_negative) {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] = seal::util::negate_uint_mod(
            seal::util::barrett_reduce_128(coeffu, coeff_modulus[j]),
            coeff_modulus[j]);
      }
    } else {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] =
            seal::util::barrett_reduce_128(coeffu, coeff_modulus[j]);
      }
    }
  } else {
    // From evaluator.h
    auto decompose_single_coeff =
        [](const seal::SEALContext::ContextData& context_data,
           const std::uint64_t* value, std::uint64_t* destination,
           seal::util::MemoryPool& pool) {
          auto& parms = context_data.parms();
          auto& coeff_modulus = parms.coeff_modulus();
          std::size_t coeff_mod_count = coeff_modulus.size();
#ifdef SEAL_DEBUG
          if (value == nullptr) {
            throw ngraph_error("value cannot be null");
          }
          if (destination == nullptr) {
            throw ngraph_error("destination cannot be null");
          }
          if (destination == value) {
            throw ngraph_error("value cannot be the same as destination");
          }
#endif
          if (coeff_mod_count == 1) {
            seal::util::set_uint_uint(value, coeff_mod_count, destination);
            return;
          }

          auto value_copy(seal::util::allocate_uint(coeff_mod_count, pool));
          for (std::size_t j = 0; j < coeff_mod_count; j++) {
            // Manually inlined for efficiency
            // Make a fresh copy of value
            seal::util::set_uint_uint(value, coeff_mod_count, value_copy.get());

            // Starting from the top, reduce always 128-bit blocks
            for (std::size_t k = coeff_mod_count - 1; k--;) {
              value_copy[k] = seal::util::barrett_reduce_128(
                  value_copy.get() + k, coeff_modulus[j]);
            }
            destination[j] = value_copy[0];
          }
        };

    // Slow case
    auto coeffu(seal::util::allocate_uint(coeff_mod_count, pool));
    auto decomp_coeffu(seal::util::allocate_uint(coeff_mod_count, pool));

    // We are at this point guaranteed to fit in the allocated space
    seal::util::set_zero_uint(coeff_mod_count, coeffu.get());
    auto coeffu_ptr = coeffu.get();
    while (coeffd >= 1) {
      *coeffu_ptr++ = static_cast<uint64_t>(fmod(coeffd, two_pow_64));
      coeffd /= two_pow_64;
    }

    // Next decompose this coefficient
    decompose_single_coeff(context_data, coeffu.get(), decomp_coeffu.get(),
                           pool);

    // Finally replace the sign if necessary
    if (is_negative) {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] =
            seal::util::negate_uint_mod(decomp_coeffu[j], coeff_modulus[j]);
      }
    } else {
      for (size_t j = 0; j < coeff_mod_count; j++) {
        destination[j] = decomp_coeffu[j];
      }
    }
  }
}

void ngraph::he::add_plain_inplace(seal::Ciphertext& encrypted, double value,
                                   const HESealBackend& he_seal_backend) {
  // Verify parameters.
  auto context = he_seal_backend.get_context();
  if (!seal::is_metadata_valid_for(encrypted, context)) {
    throw ngraph_error("encrypted is not valid for encryption parameters");
  }
  auto& context_data = *context->get_context_data(encrypted.parms_id());
  auto& parms = context_data.parms();

  NGRAPH_CHECK(parms.scheme() == seal::scheme_type::CKKS,
               "Scheme type must be CKKS");
  if (parms.scheme() == seal::scheme_type::CKKS && !encrypted.is_ntt_form()) {
    throw ngraph_error("CKKS encrypted must be in NTT form");
  }

  // Extract encryption parameters.
  auto& coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_mod_count = coeff_modulus.size();

  // Size check
  if (!seal::util::product_fits_in(coeff_count, coeff_mod_count)) {
    throw ngraph_error("invalid parameters");
  }

  NGRAPH_CHECK(encrypted.data() != nullptr, "Encrypted data == nullptr");

  // Encode
  std::vector<std::uint64_t> plaintext_vals(coeff_mod_count, 0);
  double scale = encrypted.scale();
  ngraph::he::encode(value, scale, encrypted.parms_id(), plaintext_vals,
                     he_seal_backend);

  for (size_t j = 0; j < coeff_mod_count; j++) {
    // Add poly scalar instead of poly poly
    ngraph::he::add_poly_scalar_coeffmod(
        encrypted.data() + (j * coeff_count), coeff_count, plaintext_vals[j],
        coeff_modulus[j], encrypted.data() + (j * coeff_count));
  }

#ifndef SEAL_ALLOW_TRANSPARENT_CIPHERTEXT
  // Transparent ciphertext output is not allowed.
  if (encrypted.is_transparent()) {
    throw ngraph_error("result ciphertext is transparent");
  }
#endif
}

void ngraph::he::multiply_plain_inplace(seal::Ciphertext& encrypted,
                                        double value,
                                        const HESealBackend& he_seal_backend,
                                        seal::MemoryPoolHandle pool) {
  // Verify parameters.
  auto context = he_seal_backend.get_context();
  if (!seal::is_metadata_valid_for(encrypted, context)) {
    throw ngraph_error("encrypted is not valid for encryption parameters");
  }
  if (!context->get_context_data(encrypted.parms_id())) {
    throw ngraph_error("encrypted is not valid for encryption parameters");
  }
  if (!encrypted.is_ntt_form()) {
    throw ngraph_error("encrypted is not NTT form");
  }
  if (!pool) {
    throw ngraph_error("pool is uninitialized");
  }

  // Extract encryption parameters.
  auto& context_data = *context->get_context_data(encrypted.parms_id());
  auto& parms = context_data.parms();
  auto& coeff_modulus = parms.coeff_modulus();
  size_t coeff_count = parms.poly_modulus_degree();
  size_t coeff_mod_count = coeff_modulus.size();
  size_t encrypted_ntt_size = encrypted.size();

  // Size check
  if (!seal::util::product_fits_in(encrypted_ntt_size, coeff_count,
                                   coeff_mod_count)) {
    throw ngraph_error("invalid parameters");
  }

  std::vector<std::uint64_t> plaintext_vals(coeff_mod_count, 0);
  // TODO: explore using different scales! Smaller scales might reduce # of
  // rescalings
  double scale = encrypted.scale();
  ngraph::he::encode(value, scale, encrypted.parms_id(), plaintext_vals,
                     he_seal_backend);
  double new_scale = scale * scale;
  // Check that scale is positive and not too large
  if (new_scale <= 0 || (static_cast<int>(log2(new_scale)) >=
                         context_data.total_coeff_modulus_bit_count())) {
    NGRAPH_INFO << "new_scale " << new_scale << " ("
                << static_cast<int>(log2(new_scale)) << " bits) out of bounds";
    NGRAPH_INFO << "Coeff mod bit count "
                << context_data.total_coeff_modulus_bit_count();
    throw ngraph_error("scale out of bounds");
  }

  auto& barrett64_ratio_map = he_seal_backend.barrett64_ratio_map();

  for (size_t i = 0; i < encrypted_ntt_size; i++) {
    for (size_t j = 0; j < coeff_mod_count; j++) {
      // Multiply by scalar instead of doing dyadic product
      if (coeff_modulus[j].value() < (1UL << 31)) {
        const std::uint64_t modulus_value = coeff_modulus[j].value();
        auto iter = barrett64_ratio_map.find(modulus_value);
        NGRAPH_CHECK(iter != barrett64_ratio_map.end(), "Modulus value ",
                     modulus_value, "not in Barrett64 ratio map");
        const std::uint64_t barrett_ratio = iter->second;
        ngraph::he::multiply_poly_scalar_coeffmod64(
            encrypted.data(i) + (j * coeff_count), coeff_count,
            plaintext_vals[j], modulus_value, barrett_ratio,
            encrypted.data(i) + (j * coeff_count));
      } else {
        seal::util::multiply_poly_scalar_coeffmod(
            encrypted.data(i) + (j * coeff_count), coeff_count,
            plaintext_vals[j], coeff_modulus[j],
            encrypted.data(i) + (j * coeff_count));
      }
    }
  }
  // Set the scale
  encrypted.scale() = new_scale;
}

void ngraph::he::multiply_poly_scalar_coeffmod64(
    const uint64_t* poly, size_t coeff_count, uint64_t scalar,
    const std::uint64_t modulus_value, const std::uint64_t const_ratio,
    uint64_t* result) {
  for (; coeff_count--; poly++, result++) {
    // Multiplication
    unsigned long long z = *poly * scalar;

    // Barrett base 2^64 reduction
    unsigned long long carry;
    // Carry will store the result modulo 2^64
    seal::util::multiply_uint64_hw64(z, const_ratio, &carry);
    // Barrett subtraction
    carry = z - carry * modulus_value;
    // Possible correction term
    *result =
        carry -
        (modulus_value &
         static_cast<uint64_t>(-static_cast<int64_t>(carry >= modulus_value)));
  }
}

size_t ngraph::he::match_to_smallest_chain_index(
    std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>& ciphers,
    const ngraph::he::HESealBackend& he_seal_backend) {
  size_t num_elements = ciphers.size();

  // (cipher_ind, chain_ind)
  std::pair<size_t, size_t> smallest_chain_ind{
      0, std::numeric_limits<size_t>::max()};
  for (size_t cipher_idx = 0; cipher_idx < num_elements; ++cipher_idx) {
    auto& cipher = *ciphers[cipher_idx];
    if (!cipher.known_value()) {
      size_t chain_ind = ngraph::he::get_chain_index(cipher, he_seal_backend);
      if (chain_ind < smallest_chain_ind.second) {
        smallest_chain_ind = std::make_pair(cipher_idx, chain_ind);
      }
    }
  }
  NGRAPH_CHECK(smallest_chain_ind.second != std::numeric_limits<size_t>::max());
  NGRAPH_DEBUG << "Matching to smallest chain index "
               << smallest_chain_ind.second;

  auto smallest_cipher = *ciphers[smallest_chain_ind.first];
#pragma omp parallel for
  for (size_t cipher_idx = 0; cipher_idx < num_elements; ++cipher_idx) {
    auto& cipher = *ciphers[cipher_idx];
    if (!cipher.known_value() && cipher_idx != smallest_chain_ind.second) {
      match_modulus_and_scale_inplace(smallest_cipher, cipher, he_seal_backend);
      size_t chain_ind = ngraph::he::get_chain_index(cipher, he_seal_backend);
      NGRAPH_CHECK(chain_ind == smallest_chain_ind.second, "chain_ind",
                   chain_ind, " does not match smallest ",
                   smallest_chain_ind.second);
    }
  }
  return smallest_chain_ind.second;
}
