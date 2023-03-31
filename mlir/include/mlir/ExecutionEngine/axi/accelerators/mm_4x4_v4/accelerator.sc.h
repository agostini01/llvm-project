#ifndef ACC_H
#define ACC_H

#define PE_M 4
#define PE_N 4
#define PE_K 4

#include "dma_engine.sc.h"
#define ACCNAME MM_4x4v4

// #define VERBOSE_ACC
#ifdef VERBOSE_ACC
#define ALOG(x) std::cout << x << std::endl
#else
#define ALOG(x)
#endif

// OP-Code Stuct
// 0000 : 0 = NOP;
// 0001 : 1 = read_A;
// 0010 : 2 = read_B;
// 0011 : 3 = read_A -> read_B;
// 0100 : 4 = compute_C;
// 0101 : 5 = read_A -> compute_C;
// 0110 : 6 = read_B -> compute_C;
// 0111 : 7 = read_A -> read_B -> compute_C;

// 1000 : 8 = send_C;
// 1001 : 9 = read_A -> send_C;
// 1010 : 10 = read_B -> send_C;
// 1011 : 11 = read_A -> read_B -> send_C;
// 1100 : 12 = compute_C -> send_C;
// 1101 : 13 = read_A -> compute_C -> send_C;
// 1110 : 14 = read_B -> compute_C -> send_C;
// 1111 : 15 = read_A -> read_B -> compute_C -> send_C;

struct opcode {
  unsigned int packet;
  bool read_A;
  bool read_B;
  bool compute_C;
  bool send_C;

  opcode(sc_uint<32> _packet) {
    ALOG("OPCODE: " << _packet);
    ALOG("Time: " << sc_time_stamp());
    packet = _packet;
    read_A = _packet.range(0, 0);
    read_B = _packet.range(1, 1);
    compute_C = _packet.range(2, 2);
    send_C = _packet.range(3, 3);
  }
};

struct code_extension {
  int N;
  int M;
  int K;

  code_extension(sc_uint<32> _packetA, sc_uint<32> _packetB) {
    N = _packetA.range(15, 0);
    M = _packetA.range(31, 16);
    K = _packetB.range(31, 0);

    ALOG("Time: " << sc_time_stamp());
    ALOG("N: " << N << ", M: " << M << ", K: " << K);
  }
};

SC_MODULE(ACCNAME) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_int<32> A_buffer[4096];
  sc_int<32> B_buffer[4096];
  sc_int<32> C_buffer[4096];
  sc_fifo_in<DATA> din1;
  sc_fifo_out<DATA> dout1;

  // Debug variables
  int process_blocks;
  int read_A_len;
  int read_B_len;
  int compute_C_len;
  int send_C_len;
  bool verbose;

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> send;
#else
  sc_signal<bool> compute;
  sc_signal<bool> send;
#endif

  code_extension acc_args = code_extension(0, 0);

  void Recv();

  void Compute(sc_int<32>[PE_M][PE_K], sc_int<32>[PE_K][PE_N],
               sc_int<32>[PE_M][PE_N]);

  void LoadA(sc_int<32>[PE_M][PE_K], int, int, int);

  void LoadB(sc_int<32>[PE_K][PE_N], int, int, int);

  void Store(sc_int<32>[PE_M][PE_N], int, int, int);

  void Schedule_Compute();

  void Send();

  void print_profile();

  int mul_int32(int, int);

  SC_HAS_PROCESS(ACCNAME);

  ACCNAME(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Recv, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Schedule_Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Send, clock.pos());
    reset_signal_is(reset, true);

    process_blocks = 0;
    read_A_len = 0;
    read_B_len = 0;
    compute_C_len = 0;
    send_C_len = 0;
    verbose = false;
  }
};

template <typename Integer>
void accelerator_dma_connect(ACCNAME *acc, DMA_DRIVER *dmad,
                             int _dma_input_buffer_size,
                             int _dma_output_buffer_size) {

  static sc_clock clk_fast("ClkFast", 1, SC_NS);
  static sc_signal<bool> sig_reset;
  static sc_fifo<DATA> din1("din1_fifo", _dma_input_buffer_size);
  static sc_fifo<DATA> dout1("dout1_fifo", _dma_output_buffer_size);

  acc->clock(clk_fast);
  acc->reset(sig_reset);
  acc->dout1(dout1);
  acc->din1(din1);

  dmad->clock(clk_fast);
  dmad->reset(sig_reset);
  dmad->dout1(dout1);
  dmad->din1(din1);
}

void ACCNAME::print_profile() {
  cout << "++++++++++++++++++++++++++++++++++++++++" << endl;
  cout << "Read A data_len: " << read_A_len << endl;
  cout << "Read B data_len: " << read_B_len << endl;
  cout << "MACs count: " << compute_C_len << endl;
  cout << "Send C data_len: " << send_C_len << endl;
  cout << "++++++++++++++++++++++++++++++++++++++++" << endl;
  cout << "Executed with :" << __FILE__ << endl;
  cout << "- - - - - - - - - - - - - - - - - - - - " << endl;
}

void ACCNAME::Recv() {

  wait();
  while (1) {

    opcode packet(din1.read().data);
    code_extension op_args(din1.read().data, din1.read().data);
    acc_args = op_args;

    if (packet.read_A) {
      int read_length = op_args.M * op_args.K;
      for (int i = 0; i < read_length; i++) {
        A_buffer[i] = din1.read().data;
        DWAIT();
      }
    }

    if (packet.read_B) {
      int read_length = op_args.K * op_args.N;
      for (int i = 0; i < read_length; i++) {
        B_buffer[i] = din1.read().data;
        DWAIT();
      }
    }

    // Computes C if true
    if (packet.compute_C) {
      compute.write(true);
      wait();
    }

    while (compute)
      wait();

    // Sends then clears C if true
    if (packet.send_C) {
      send.write(true);
      wait();
    }

    while (send)
      wait();

    wait();
  }
}

void ACCNAME::LoadA(sc_int<32> A[PE_M][PE_K], int M, int K, int in_stride) {
  //#pragma HLS inline OFF
  for (int m = 0; m < PE_M; m++) {
    for (int k = 0; k < PE_K; k++) {
      A[m][k] = A_buffer[(M + m) * in_stride + K + k];
    }
  }
}

void ACCNAME::LoadB(sc_int<32> B[PE_K][PE_N], int N, int K, int in_stride) {
  //#pragma HLS inline OFF
  for (int n = 0; n < PE_N; n++) {
    for (int k = 0; k < PE_K; k++) {
      B[k][n] = B_buffer[(K + k) * in_stride + N + n];
    }
  }
}

void ACCNAME::Compute(sc_int<32> A[PE_M][PE_K], sc_int<32> B[PE_K][PE_N],
                      sc_int<32> C[PE_M][PE_N]) {
  //#pragma HLS inline OFF
  for (int m = 0; m < PE_M; m++) {
    // #pragma HLS pipeline
    for (int n = 0; n < PE_N; n++) {
      int acc = 0;
      for (int k = 0; k < PE_K; k++) {
        int x = A[m][k];
        int y = B[k][n];
        acc += mul_int32(x, y);
      }
      C[m][n] = acc;
    }
  }
}

void ACCNAME::Store(sc_int<32> C[PE_M][PE_N], int N, int M, int out_stride) {
  //#pragma HLS inline OFF
  // #pragma HLS pipeline
  for (int m = 0; m < PE_M; m++) {
    for (int n = 0; n < PE_N; n++) {
      int C_dex = (N + n) * out_stride + M + m;
      C_buffer[C_dex] += C[m][n];
    }
  }
}

void ACCNAME::Schedule_Compute() {
  sc_int<32> A[PE_M][PE_K];
  sc_int<32> B[PE_K][PE_N];
  sc_int<32> C[PE_M][PE_N];
  // #pragma HLS array_partition variable = A complete dim = 2
  // #pragma HLS array_partition variable = B complete dim = 0
  // #pragma HLS array_partition variable = C complete dim = 2

  wait();
  while (1) {
    while (!compute)
      wait();

    for (int k = 0; k < acc_args.K; k += PE_K) {
      for (int m = 0; m < acc_args.M; m += PE_M) {
        LoadA(A, m, k, acc_args.K);
        for (int n = 0; n < acc_args.N; n += PE_N) {
          LoadB(B, n, k, acc_args.K);
          Compute(A, B, C);
          Store(C, n, m, acc_args.N);
        }
      }
    }

    wait();
    compute.write(false);
    wait();
  }
}

void ACCNAME::Send() {
  wait();
  while (1) {
    while (!send)
      wait();

    for (int m = 0; m < acc_args.M; m++) {
      for (int n = 0; n < acc_args.N; n++) {
        DATA d;
        d.tlast = false;
        d.data = C_buffer[n * acc_args.M + m];
        if (n + 1 == acc_args.N && m + 1 == acc_args.M)
          d.tlast = true;
        dout1.write(d);
        wait();
        C_buffer[n * acc_args.M + m] = 0;
        DWAIT();
      }
    }
    send.write(false);
    wait();
  }
}

int ACCNAME::mul_int32(int x, int y) { return x * y; }

#endif
