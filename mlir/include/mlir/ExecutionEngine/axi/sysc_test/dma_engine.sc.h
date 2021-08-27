#ifndef DRIVER_H
#define DRIVER_H

#include "accelerator.sc.h"

SC_MODULE(DMA_DRIVER) {
    sc_in<bool> clock;
    sc_in <bool>  reset;
    sc_fifo_in<DATA> dout1;
    sc_fifo_out<DATA> din1;
    bool sent;

    void DMA_MMS2(){
        while(1){
            for(int i=0;i<input_len;i++){
                int d = DMA_input_buffer[i];
                din1.write({d, 1});
                wait();
            }
            
            sent = true;
            while(sent)wait();
            wait();
            sc_pause();
            wait();
	    }
        
    };

    void DMA_S2MM(){
        while(1){
            while(!sent)wait();
            bool last = false;
            int i=0;
            do{
                DATA d = dout1.read();
                last = d.tlast;
                DMA_output_buffer[i++]  = d.data;
                wait();
            } while (!last);
            sent=false;
            wait();
        }
    };

    SC_HAS_PROCESS(DMA_DRIVER);
    
    DMA_DRIVER(sc_module_name name_) :sc_module(name_) {
        SC_CTHREAD(DMA_MMS2, clock.pos());
        reset_signal_is(reset,true);

        SC_CTHREAD(DMA_S2MM, clock.pos());
        reset_signal_is(reset,true);

    }

    int* DMA_input_buffer;
    int* DMA_output_buffer;
    int input_len;
};


#endif