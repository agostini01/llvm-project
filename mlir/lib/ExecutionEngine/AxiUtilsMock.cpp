//===- AxiUtils.cpp - AXI4MLIR  implementation ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements wrapper AXI4MLIR library calls. These are the calls
// visible to the MLIR.
//
// This is a mock implementation that only prints to the terminal.
//
//===----------------------------------------------------------------------===//

#include "mlir/ExecutionEngine/AxiUtils.h"

// =============================================================================
// AXI_APIV1
// =============================================================================

extern "C" void dma_init(unsigned int dma_address,
                         unsigned int dma_input_address,
                         unsigned int dma_input_buffer_size,
                         unsigned int dma_output_address,
                         unsigned int dma_output_buffer_size) {
  std::cout << "Called: " << __func__ << std::endl;
  std::cout << "\t" << dma_address << std::endl;
  std::cout << "\t" << dma_input_address << std::endl;
  std::cout << "\t" << dma_input_buffer_size << std::endl;
  std::cout << "\t" << dma_output_address << std::endl;
  std::cout << "\t" << dma_output_buffer_size << std::endl;
  std::cout << "Called: " << __func__ << std::endl;
  return;
}

extern "C" void dma_free() { std::cout << "Called: " << __func__ << std::endl; }

extern "C" unsigned int *dma_get_inbuffer() {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" unsigned int *dma_get_outbuffer() {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" int dma_copy_to_inbuffer(unsigned int *host_src_address,
                                    int data_length, int offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" int dma_copy_from_outbuffer(unsigned int *host_dst_address,
                                       int data_length, int offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" int dma_start_send(int length, int offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" int dma_check_send() {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" void dma_wait_send() {
  std::cout << "Called: " << __func__ << std::endl;
}

extern "C" int dma_start_recv(int length, int offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" void dma_wait_recv() {
  std::cout << "Called: " << __func__ << std::endl;
}

extern "C" int dma_check_recv() {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" unsigned int dma_set(unsigned int *dma_virtual_address, int offset,
                                unsigned int value) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" unsigned int dma_get(unsigned int *dma_virtual_address, int offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}
