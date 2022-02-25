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

extern "C" void dma_init(uint64_t dma_address,
                         uint64_t dma_input_address,
                         uint64_t dma_input_buffer_size,
                         uint64_t dma_output_address,
                         uint64_t dma_output_buffer_size) {
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

extern "C" uint64_t *dma_get_inbuffer() {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" uint64_t *dma_get_outbuffer() {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" int64_t dma_copy_to_inbuffer(uint64_t *host_src_address,
                                    int64_t data_length, int64_t offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" int64_t dma_copy_from_outbuffer(uint64_t *host_dst_address,
                                       int64_t data_length, int64_t offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

template <typename T>
int64_t mlir_dma_copy_to_inbuffer(const DynamicMemRefType<T> &src, int64_t data_length,
                              int64_t offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" int64_t _mlir_ciface_copy_to_inbuffer_f32(UnrankedMemRefType<float> *M,
                                                 int64_t offset) {
  mlir_dma_copy_to_inbuffer(DynamicMemRefType<float>(*M), 0, offset);
  return 0;
}

extern "C" int64_t copy_to_inbuffer_f32(int64_t rank, void *ptr, int64_t offset) {
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  return 0;
}

extern "C" int64_t copy_to_inbuffer_i32(int64_t rank, void *ptr, int64_t offset) {
  UnrankedMemRefType<int64_t> descriptor = {rank, ptr};
  return 0;
}

extern "C" int64_t copy_from_outbuffer_f32(int64_t rank, void *ptr,
                                       int64_t offset) {
  UnrankedMemRefType<float> descriptor = {rank, ptr};
  return 0;
}

extern "C" int64_t dma_start_send(int64_t length, int64_t offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" int64_t dma_check_send() {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" void dma_wait_send() {
  std::cout << "Called: " << __func__ << std::endl;
}

extern "C" int64_t dma_start_recv(int64_t length, int64_t offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" void dma_wait_recv() {
  std::cout << "Called: " << __func__ << std::endl;
}

extern "C" int64_t dma_check_recv() {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" uint64_t dma_set(uint64_t *dma_virtual_address, int64_t offset,
                                uint64_t value) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}

extern "C" uint64_t dma_get(uint64_t *dma_virtual_address, int64_t offset) {
  std::cout << "Called: " << __func__ << std::endl;
  return 0;
}
