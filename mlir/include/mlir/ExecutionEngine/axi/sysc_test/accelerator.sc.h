#ifndef GEMM_H
#define GEMM_H

#include <systemc.h>

typedef struct _DATA {
  sc_uint<32> data;
  bool tlast;

  inline friend ostream &operator<<(ostream &os, const _DATA &v) {
    cout << "data&colon; " << v.data << " tlast: " << v.tlast;
    return os;
  }
} DATA;

SC_MODULE(MMAcc) {
  sc_in<bool> clock;
  sc_in<bool> reset;
  sc_int<32> inputs[4096];
  sc_int<32> weights[4096];
  sc_int<32> outputs[4096];
  sc_fifo_in<DATA> din1;
  sc_fifo_out<DATA> dout1;

#ifndef __SYNTHESIS__
  sc_signal<bool, SC_MANY_WRITERS> compute;
  sc_signal<bool, SC_MANY_WRITERS> send;
#else
  sc_signal<bool> compute;
  sc_signal<bool> send;
#endif

  void Recv() {
    while (1) {
      while (compute)
        wait();
      for (int i = 0; i < 16; i++)
        inputs[i] = din1.read().data;
      for (int i = 0; i < 16; i++)
        weights[i] = din1.read().data;
      wait();
      compute.write(true);
      wait();
    }
  }

  void Compute() {
    wait();
    while (1) {
      while (!compute)
        wait();
      for (int i = 0; i < 4; i++) {
        for (int w = 0; w < 4; w++) {
          int acc = 0;
          for (int d = 0; d < 4; d++) {
            int x = inputs[i * 4 + d];
            int y = weights[w * 4 + d];
            acc += x * y;
          }
          outputs[i * 4 + w] = acc;
        }
      }
      wait();
      compute.write(false);
      send.write(true);
      wait();
    }
  }

  void Send() {
    DATA last = {0, 1};
    wait();
    while (1) {
      while (!send)
        wait();
      for (int i = 0; i < 16; i++) {
        DATA d;
        d.tlast = false;
        if (i == 15)
          d.tlast = true;
        d.data = outputs[i];
        dout1.write(d);
      }
      send.write(false);
      wait();
    }
  }

  SC_HAS_PROCESS(MMAcc);

  MMAcc(sc_module_name name_) : sc_module(name_) {
    SC_CTHREAD(Recv, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Compute, clock.pos());
    reset_signal_is(reset, true);

    SC_CTHREAD(Send, clock.pos());
    reset_signal_is(reset, true);

    // #pragma HLS RESOURCE variable=din1 core=AXI4Stream metadata="-bus_bundle
    // S_AXIS_DATA1" port_map={{din1_0 TDATA} {din1_1 TLAST}} #pragma HLS
    // RESOURCE variable=dout1 core=AXI4Stream metadata="-bus_bundle
    // M_AXIS_DATA1" port_map={{dout1_0 TDATA} {dout1_1 TLAST}} #pragma HLS
    // RESET variable=reset
  }
};
#endif