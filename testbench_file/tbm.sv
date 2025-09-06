`timescale 1ns / 1ps
module tb_mnist_accel;
    reg clk, reset, start;
    reg signed [31:0] image_pixels [0:783];

    wire [3:0] predicted;
    mnist_accel dut (
        .clk(clk), .reset(reset), .start(start),
        .image_pixels(image_pixels),
        .predicted_digit(predicted)
    );

    initial begin
        clk = 0; forever #5 clk = ~clk;
    end

    initial begin
        reset = 1; start = 0;
        #20; reset = 0;

        // Preload image and the (small) layer2 weight files + biases:
        $readmemh("image_0000.mem", image_pixels);
        $readmemh("W2_flat.mem", dut.W2_flat); // make sure this file has OUT2*HIDDEN lines
        $readmemh("B1.mem", dut.B1);           // B1 is used only for matvec_rombank biases
        $readmemh("B2.mem", dut.B2);

        // Note: W1 is loaded by rom_neuron initial blocks from W1_neuron_*.mem files

        #10 start = 1;
        #10 start = 0;
        // Wait long enough for the two matvec computations to finish (depends on clock)
        #200000;
        $display("Predicted digit = %d", predicted);
        $finish;
    end
endmodule
