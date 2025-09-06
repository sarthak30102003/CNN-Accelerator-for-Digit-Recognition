// matvec.v - Fully-connected layer (matrix-vector multiply + bias + optional ReLU)
module matvec #(
    parameter IN_DIM = 784,      // input dimension (e.g. 784 for 28x28 image)
    parameter OUT_DIM = 10,      // output dimension (e.g. 10 classes or hidden neurons)
    parameter WIDTH = 32,        // data width (Q8.24 fixed-point)
    parameter FRAC_BITS = 24     // fractional bits in Q format
)(
    input clk,
    input reset,
    input start,
    // Input vector (should be loaded beforehand)
    input signed [WIDTH-1:0] in_vec [0:IN_DIM-1],
    // Weights and biases (pre-loaded or assigned)
    input signed [WIDTH-1:0] weight [0:OUT_DIM-1][0:IN_DIM-1],
    input signed [WIDTH-1:0] bias  [0:OUT_DIM-1],
    output reg signed [WIDTH-1:0] out_vec [0:OUT_DIM-1],
    output reg done
);
    // State machine indices
    integer i, j;
    reg signed [WIDTH*2-1:0] prod;   // 64-bit product accumulator (since WIDTH=32, product 64-bit)
    reg signed [WIDTH-1:0] acc;      // 32-bit accumulator for each neuron sum

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            i <= 0; j <= 0;
            acc <= 0;
            done <= 0;
        end else if (start) begin
            done <= 0;
            i <= 0;
            j <= 0;
            acc <= bias[0];          // initialize sum with bias of neuron 0
        end else if (!done) begin
            if (i < OUT_DIM) begin
                if (j < IN_DIM) begin
                    // Multiply input * weight (signed) and align fraction
                    prod = in_vec[j] * weight[i][j];
                    // Accumulate (arithmetic right shift aligns Q8.24)
                    acc = acc + (prod >>> FRAC_BITS);
                    j = j + 1;
                end else begin
                    // Finished all inputs for neuron i; store result
                    out_vec[i] = acc;
                    // Move to next neuron
                    i = i + 1;
                    if (i < OUT_DIM) begin
                        acc = bias[i];     // start next neuron sum with bias
                        j = 0;
                    end else begin
                        done = 1;          // all neurons done
                    end
                end
            end
        end
    end
endmodule
