#ifndef AXI_APIv1
#define AXI_APIv1

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef SYSC
// Easy way to switch between systemC accelerators --- there is probably a better way
#ifdef ACC_V4
#include "mlir/ExecutionEngine/axi/accelerators/mm_4x4_v4/accelerator.sc.h"
#elif  ACC_V3
#include "mlir/ExecutionEngine/axi/accelerators/mm_4x4_v3/accelerator.sc.h"
#elif  ACC_V2
#include "mlir/ExecutionEngine/axi/accelerators/mm_4x4_v2/accelerator.sc.h"
#else
#include "mlir/ExecutionEngine/axi/accelerators/mm_4x4_v1/accelerator.sc.h"
#endif
#endif

// API Model = One DMA is allocated with a single input and output buffer (Can
// have different size)

// Simple view of DMA
/*
dma -> {
    control_register_address : uint64_t     # Mapped address to the start of
the DMA control registers Buffer input_buffer (address,size) Buffer
output_buffer  (address,size)
}
*/

struct dma {
#define MM2S_CONTROL_REGISTER 0x00
#define MM2S_STATUS_REGISTER 0x04
#define MM2S_START_ADDRESS 0x18
#define MM2S_LENGTH 0x28
#define S2MM_CONTROL_REGISTER 0x30
#define S2MM_STATUS_REGISTER 0x34
#define S2MM_DESTINATION_ADDRESS 0x48
#define S2MM_LENGTH 0x58
#define PAGE_SIZE getpagesize()

#define m_assert(expr, msg) assert(((void)(msg), (expr)))

#define PROFILE
#ifdef PROFILE
#define PLOG(x) std::cout << x << std::endl
#define PFUNC(x) x
#else
#define PLOG(x)
#define PFUNC(x)
#endif

// #define VERBOSE_AXI
#ifdef VERBOSE_AXI
// #define LOG(x) std::cout << x << std::endl
#define LOG(x)
#else
#define LOG(x)
#endif

  uint64_t *dma_address;
  uint64_t *dma_input_address;
  uint64_t *dma_output_address;
  uint64_t dma_input_buffer_size;
  uint64_t dma_output_buffer_size;
  uint64_t dma_input_paddress;
  uint64_t dma_output_paddress;
  uint64_t *acc_address;
  uint64_t current_input_offset;

  // Profiling Variables
  uint64_t dma_send_length=0;
  uint64_t dma_recv_length=0;
  uint64_t dma_send_count=0;
  uint64_t dma_recv_count=0;


  // temp --- need to remove later
  bool verbose;

#ifdef SYSC
  ACCNAME *acc;
  DMA_DRIVER *dmad;
#endif

  void dma_init(uint64_t dma_address, uint64_t dma_input_address,
                uint64_t dma_input_buffer_size,
                uint64_t dma_output_address,
                uint64_t dma_output_buffer_size);

  // Memory unmaps DMA control_register_address and Input and output buffers
  void dma_free();

  // We could reduce to one set of the following calls
  //================================================================================================================

  //-----------------BUFFER Functions-----------------
  // Get the MMap address of the input buffer of the dma  *Needed to copy data
  // to Input_Buffer*
  uint64_t *dma_get_inbuffer();

  // Get the MMap address of the output buffer of the dma *Needed to copy data
  // from Output_Buffer*
  uint64_t *dma_get_outbuffer();

  //================================================================================================================

  //-----------------BUFFER Functions-----------------
  // Copy data into the Input Buffer (length to write, offset to write to)
  // returns 0 if successful
  int64_t dma_copy_to_inbuffer(uint64_t *host_src_address, int64_t data_length,
                           int64_t offset);

  template <typename T>
  int64_t mlir_dma_copy_to_inbuffer(T *mr_base, int64_t mr_dim, int64_t mr_rank,
                                int64_t mr_offset, const int64_t *mr_sizes,
                                const int64_t *mr_strides, int64_t dma_offset);

  // Copy data from the Output Buffer (length to read, offset to read from)
  // returns 0 if successful
  int64_t dma_copy_from_outbuffer(uint64_t *host_dst_address, int64_t data_length,
                              int64_t offset);

  template <typename T>
  int64_t mlir_dma_copy_from_outbuffer(T *mr_base, int64_t mr_dim, int64_t mr_rank,
                                   int64_t mr_offset, const int64_t *mr_sizes,
                                   const int64_t *mr_strides, int64_t dma_offset);

  //============================================================================

  //-----------------DMA MMS2 Functions-----------------
  /**
   * Checks if input buffer size is >= length
   * Sets DMA MMS2 transfer length to length
   * Starts transfers to the accelerator using dma associated with dma_id
   * Return 0 if successful, returns negative if error occurs
   */
  int64_t dma_start_send(int64_t length, int64_t offset);

  // Blocks thread until dma MMS2 transfer is complete
  void dma_wait_send();

  // Same as dma_send but thread does not block, returns 0 if done
  int64_t dma_check_send();

  //-----------------DMA S2MM Functions-----------------
  /**
   * Checks if buffer size is >= length
   * Sets 2SMM store length
   * Starts storing data recieved through dma associated with dma_id
   * Return 0 if successful, returns negative if error occurs
   */
  int64_t dma_start_recv(int64_t length, int64_t offset);

  // Blocks thread until dma S2MM transfer is complete (TLAST signal is seen)
  void dma_wait_recv();

  // Same as dma_recv but thread does not block, returns 0 if done
  int64_t dma_check_recv();

  //********************************** Unexposed Functions
  //**********************************
  void initDMAControls();
  void dma_set(uint64_t *dma_virtual_address, int64_t offset,
               uint64_t value);
  uint64_t dma_get(uint64_t *dma_virtual_address, int64_t offset);
  void dma_mm2s_sync();
  void dma_s2mm_sync();
  void acc_init(uint64_t base_addr, int64_t length);
  void dump_acc_signals(int64_t state);
};

#endif