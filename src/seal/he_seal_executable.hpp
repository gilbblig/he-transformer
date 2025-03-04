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

#include <atomic>
#include <boost/asio.hpp>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "he_tensor.hpp"
#include "ngraph/runtime/backend.hpp"
#include "ngraph/util.hpp"
#include "node_wrapper.hpp"
#include "seal/he_seal_backend.hpp"
#include "seal/seal.h"
#include "seal/seal_ciphertext_wrapper.hpp"
#include "tcp/tcp_message.hpp"
#include "tcp/tcp_session.hpp"

using boost::asio::ip::tcp;

namespace ngraph {
namespace he {
class HESealExecutable : public runtime::Executable {
 public:
  HESealExecutable(const std::shared_ptr<Function>& function,
                   bool enable_performance_collection,
                   ngraph::he::HESealBackend& he_seal_backend,
                   bool encrypt_data, bool encrypt_model, bool batch_data,
                   bool complex_packing);

  ~HESealExecutable() {
    if (m_enable_client) {
      // Wait until thread finishes with m_io_context
      m_thread.join();

      // TODO: why is this needed to prevent m_acceptor from double-freeing?

      // m_acceptor and m_io_context both free the socket? so avoid double-free
      m_acceptor->close();
      m_acceptor = nullptr;
      m_session = nullptr;
    }
  }

  void client_setup();

  void start_server();

  bool call(
      const std::vector<std::shared_ptr<runtime::Tensor>>& outputs,
      const std::vector<std::shared_ptr<runtime::Tensor>>& inputs) override;

  std::vector<runtime::PerformanceCounter> get_performance_data()
      const override;

  size_t get_port() const { return m_port; };

  // TODO: merge _done() methods
  bool relu_done() const { return m_relu_done; };
  bool max_done() const { return m_max_done; };
  bool minimum_done() const { return m_minimum_done; };

  bool session_started() const { return m_session_started; };

  bool client_inputs_received() const { return m_client_inputs_received; };

  void accept_connection();

  void check_client_supports_function();

  void handle_message(const TCPMessage& message);

  void handle_server_relu_op(std::shared_ptr<HESealCipherTensor>& arg0_cipher,
                             std::shared_ptr<HESealCipherTensor>& out_cipher,
                             const NodeWrapper& node_wrapper);

  bool verbose_op(const ngraph::Node& op) {
    return m_verbose_all_ops ||
           m_verbose_ops.find(ngraph::to_lower(op.description())) !=
               m_verbose_ops.end();
  };

  void enable_client() {
    m_enable_client = true;
    client_setup();
  }

 private:
  HESealBackend& m_he_seal_backend;
  bool m_encrypt_data;
  bool m_encrypt_model;
  bool m_batch_data;
  bool m_is_compiled;
  bool m_complex_packing;
  bool m_verbose_all_ops;

  bool m_enable_client;
  size_t m_batch_size;
  size_t m_port;  // Which port the server is hosted at

  std::unordered_map<std::shared_ptr<const Node>, stopwatch> m_timer_map;
  std::vector<NodeWrapper> m_wrapped_nodes;

  std::unique_ptr<tcp::acceptor> m_acceptor;

  // Must be shared, since TCPSession uses enable_shared_from_this()
  std::shared_ptr<TCPSession> m_session;
  std::thread m_thread;
  boost::asio::io_context m_io_context;

  // (Encrypted) inputs to compiled function
  std::vector<std::shared_ptr<ngraph::he::HETensor>> m_client_inputs;
  // (Encrypted) outputs of compiled function
  std::vector<std::shared_ptr<ngraph::he::HETensor>> m_client_outputs;

  std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>
      m_relu_ciphertexts;
  std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>
      m_max_ciphertexts;
  std::vector<std::shared_ptr<ngraph::he::SealCiphertextWrapper>>
      m_minimum_ciphertexts;

  std::set<std::string> m_verbose_ops;

  std::shared_ptr<seal::SEALContext> m_context;

  // To trigger when relu is done
  std::mutex m_relu_mutex;
  std::condition_variable m_relu_cond;
  bool m_relu_done;
  std::vector<size_t> m_unknown_relu_idx;

  // To trigger when maxpool is done
  std::mutex m_max_mutex;
  std::condition_variable m_max_cond;
  bool m_max_done;

  // To trigger when minimum is done
  std::mutex m_minimum_mutex;
  std::condition_variable m_minimum_cond;
  bool m_minimum_done;

  // To trigger when result message has been written
  std::mutex m_result_mutex;
  std::condition_variable m_result_cond;
  bool m_result_done;

  // To trigger when session has started
  std::mutex m_session_mutex;
  std::condition_variable m_session_cond;
  bool m_session_started;

  // To trigger when client inputs have been received
  std::mutex m_client_inputs_mutex;
  std::condition_variable m_client_inputs_cond;
  bool m_client_inputs_received;

  void generate_calls(const element::Type& type, const NodeWrapper& op,
                      const std::vector<std::shared_ptr<HETensor>>& outputs,
                      const std::vector<std::shared_ptr<HETensor>>& inputs);
};
}  // namespace he
}  // namespace ngraph
