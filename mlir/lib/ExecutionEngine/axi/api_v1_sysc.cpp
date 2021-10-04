//===- api_v1.cpp - AXI core API implementation ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the core functions to use the AXI DMA interface.
//
//===----------------------------------------------------------------------===//

#define SYSC
#include "mlir/ExecutionEngine/axi/api_v1.h"

int sc_main(int argc, char *argv[]) { return 0; }

// SystemC code does not require all these parameters
void dma::dma_init(unsigned int _dma_address, unsigned int _dma_input_address,
                   unsigned int _dma_input_buffer_size,
                   unsigned int _dma_output_address,
                   unsigned int _dma_output_buffer_size) {

  sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", SC_DO_NOTHING);
  sc_report_handler::set_actions(SC_ID_LOGIC_X_TO_BOOL_, SC_LOG);
  sc_report_handler::set_actions(SC_ID_VECTOR_CONTAINS_LOGIC_VALUE_, SC_LOG);

  dma_input_address =
      (unsigned int *)malloc(_dma_input_buffer_size * sizeof(int));
  dma_output_address =
      (unsigned int *)malloc(_dma_output_buffer_size * sizeof(int));

  // Initialize with zeros
  for (int64_t i = 0; i < _dma_input_buffer_size; i++) {
    *(dma_input_address + i) = 0;
  }

  for (int64_t i = 0; i < _dma_output_buffer_size; i++) {
    *(dma_output_address + i) = 0;
  }

  static ACCNAME dut("dut");
  static DMA_DRIVER dm("DMA");
  accelerator_dma_connect<int>(&dut, &dm, _dma_input_buffer_size,
                               _dma_output_buffer_size);

  dm.DMA_input_buffer = (int *)dma_input_address;
  dm.DMA_output_buffer = (int *)dma_output_address;

  acc = &dut;
  dmad = &dm;
  LOG("SystemC dma_init() initializes the DMA");
}

void dma::dma_free() {
  LOG("SystemC dma_free() deallocates DMA buffers");
  free(dma_input_address);
  free(dma_output_address);
}

unsigned int *dma::dma_get_inbuffer() { return dma_input_address; }

unsigned int *dma::dma_get_outbuffer() { return dma_output_address; }

int dma::dma_copy_to_inbuffer(unsigned int *src_address, int data_length,
                              int offset) {
  LOG("SystemC dma_copy_to_inbuffer()");
  m_assert("data copy will overflow input buffer",
           (unsigned int)(offset + data_length) <= dma_input_buffer_size);
  memcpy((dma_get_inbuffer() + offset), src_address, data_length * 4);
  return 0;
}

int dma::dma_copy_from_outbuffer(unsigned int *dst_address, int data_length,
                                 int offset) {
  LOG("SystemC dma_copy_from_outbuffer()");
  m_assert("tries to access data out with the output buffer",
           (unsigned int)(offset + data_length) <= dma_output_buffer_size);
  memcpy(dst_address, (dma_get_outbuffer() + offset), data_length * 4);
  return 0;
}

// Implements the actual copy
template <typename T>
int dma::mlir_dma_copy_to_inbuffer(T *mr_base, int64_t mr_dim, int64_t mr_rank,
                                   int64_t mr_offset, const int64_t *mr_sizes,
                                   const int64_t *mr_strides, int dma_offset) {
  std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "]\n";

  int64_t total_size = 1;
  for (int64_t i = 0; i < mr_rank; i++) {
    total_size *= mr_sizes[i];
  }

  for (int64_t i = 0; i < total_size; i++) {
    *(dma_get_inbuffer() + dma_offset + i) = mr_base[i + mr_offset];
  }

  return 0;
}

template <typename T>
int dma::mlir_dma_copy_from_outbuffer(T *mr_base, int64_t mr_dim,
                                      int64_t mr_rank, int64_t mr_offset,
                                      const int64_t *mr_sizes,
                                      const int64_t *mr_strides,
                                      int dma_offset) {

  std::cout << __FILE__ << ": " << __LINE__ << " [" << __func__ << "]\n";

  int64_t total_size = 1;
  for (int64_t i = 0; i < mr_rank; i++) {
    total_size *= mr_sizes[i];
  }

  for (int64_t i = 0; i < total_size; i++) {
    mr_base[i + mr_offset] = *(dma_get_outbuffer() + dma_offset + i);
  }

  return 0;
}

// Make templates concrete:
template int dma::mlir_dma_copy_to_inbuffer<float>(
    float *mr_base, int64_t mr_dim, int64_t mr_rank, int64_t mr_offset,
    const int64_t *mr_sizes, const int64_t *mr_strides, int dma_offset);

template int dma::mlir_dma_copy_from_outbuffer<float>(
    float *mr_base, int64_t mr_dim, int64_t mr_rank, int64_t mr_offset,
    const int64_t *mr_sizes, const int64_t *mr_strides, int dma_offset);

int dma::dma_start_send(int length, int offset) {
  LOG("SystemC dma_start_send()");
  dmad->input_len = length;
  dmad->input_offset = offset;
  dmad->send = true;
  return 0;
}

void dma::dma_wait_send() {
  LOG("SystemC dma_wait_send() starts simulation");
  sc_start();
}

int dma::dma_check_send() {
  LOG("SystemC dma_check_send() does nothing");
  return 0;
}

int dma::dma_start_recv(int length, int offset) {
  LOG("SystemC dma_start_recv()");
  dmad->output_len = length;
  dmad->output_offset = offset;
  dmad->recv = true;
  return 0;
}

void dma::dma_wait_recv() {
  LOG("SystemC dma_wait_recv() starts simulation");
  sc_start();
}

int dma::dma_check_recv() {
  LOG("SystemC dma_check_recv() does nothing");
  return 0;
}

// We really don't need any of the functions below to be implemented for SystemC
//********************************** Unexposed Functions
//**********************************
void dma::initDMAControls() {
  dma_set(dma_address, S2MM_CONTROL_REGISTER, 4);
  dma_set(dma_address, MM2S_CONTROL_REGISTER, 4);
  dma_set(dma_address, S2MM_CONTROL_REGISTER, 0);
  dma_set(dma_address, MM2S_CONTROL_REGISTER, 0);
  dma_set(dma_address, S2MM_DESTINATION_ADDRESS,
          (unsigned long)dma_output_address); // Write destination address
  dma_set(dma_address, MM2S_START_ADDRESS,
          (unsigned long)dma_input_address); // Write source address
  dma_set(dma_address, S2MM_CONTROL_REGISTER, 0xf001);
  dma_set(dma_address, MM2S_CONTROL_REGISTER, 0xf001);
}

void dma::dma_set(unsigned int *dma_address, int offset, unsigned int value) {
  dma_address[offset >> 2] = value;
}

unsigned int dma::dma_get(unsigned int *dma_address, int offset) {
  return dma_address[offset >> 2];
}

void dma::dma_mm2s_sync() {
  msync(dma_address, PAGE_SIZE, MS_SYNC);
  unsigned int mm2s_status = dma_get(dma_address, MM2S_STATUS_REGISTER);
  while (!(mm2s_status & 1 << 12) || !(mm2s_status & 1 << 1)) {
    msync(dma_address, PAGE_SIZE, MS_SYNC);
    mm2s_status = dma_get(dma_address, MM2S_STATUS_REGISTER);
  }
}

void dma::dma_s2mm_sync() {
  msync(dma_address, PAGE_SIZE, MS_SYNC);
  unsigned int s2mm_status = dma_get(dma_address, S2MM_STATUS_REGISTER);
  while (!(s2mm_status & 1 << 12) || !(s2mm_status & 1 << 1)) {
    msync(dma_address, PAGE_SIZE, MS_SYNC);
    s2mm_status = dma_get(dma_address, S2MM_STATUS_REGISTER);
  }
}

void dma::acc_init(unsigned int base_addr, int length) {
  int dh = open("/dev/mem", O_RDWR | O_SYNC);
  size_t virt_base = base_addr & ~(PAGE_SIZE - 1);
  size_t virt_offset = base_addr - virt_base;
  void *addr = mmap(NULL, length + virt_offset, PROT_READ | PROT_WRITE,
                    MAP_SHARED, dh, virt_base);
  close(dh);
  if (addr == (void *)-1)
    exit(EXIT_FAILURE);
  acc_address = reinterpret_cast<unsigned int *>(addr);
}

void dma::dump_acc_signals(int state) {
  msync(acc_address, PAGE_SIZE, MS_SYNC);
  std::ofstream file;
  file.open("dump_acc_signals.dat", std::ios_base::app);
  file << "====================================================" << std::endl;
  file << "State: " << state << std::endl;
  file << "====================================================" << std::endl;
  for (int i = 0; i < 16; i++)
    file << acc_address[i] << ",";
  file << "====================================================" << std::endl;
}