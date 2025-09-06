// mnist_accel.v - Top-level MLP accelerator for MNIST
module mnist_accel(
    input clk,
    input reset,
    input start,  // pulse to start inference
    // Input image pixels (28x28 = 784, Q8.24 each)
    input  signed [31:0] image_pixels [0:783],
    output reg [3:0] predicted_digit  // 4-bit output class label
);
    // Parameters: define hidden layer size
    parameter HIDDEN = 128;  // example hidden size

    // Internal signals
    reg layer1_start, layer2_start;
    wire layer1_done, layer2_done;
    wire signed [31:0] layer1_out [0:HIDDEN-1];
    wire signed [31:0] layer2_out [0:9];

    // Weights and biases (for brevity, declared as memories; to be initialized)
    reg signed [31:0] W1 [0:HIDDEN-1][0:783];
    reg signed [31:0] B1 [0:HIDDEN-1];
    reg signed [31:0] W2 [0:9][0:HIDDEN-1];
    reg signed [31:0] B2 [0:9];

    // Instantiate first layer (784 -> HIDDEN, with ReLU)
    matvec #(.IN_DIM(784), .OUT_DIM(HIDDEN)) layer1 (
        .clk(clk), .reset(reset), .start(layer1_start),
        .in_vec(image_pixels),
        .weight(W1), .bias(B1),
        .out_vec(layer1_out),
        .done(layer1_done)
    );
    // Apply ReLU in-place (could be done inside matvec or here)
    genvar n;
    generate
      for (n = 0; n < HIDDEN; n = n + 1) begin: relu_hidden
        wire signed [31:0] relu_val;
        relu relu_inst(.din(layer1_out[n]), .dout(relu_val));
        // Register or wire this if needed; here we feed directly to layer2
        assign layer1_out[n] = relu_val;
      end
    endgenerate

    // Instantiate second layer (HIDDEN -> 10, no ReLU on output, we just take raw scores)
    matvec #(.IN_DIM(HIDDEN), .OUT_DIM(10)) layer2 (
        .clk(clk), .reset(reset), .start(layer2_start),
        .in_vec(layer1_out),
        .weight(W2), .bias(B2),
        .out_vec(layer2_out),
        .done(layer2_done)
    );

    // Argmax on final scores
    argmax classify(
        .in0(layer2_out[0]), .in1(layer2_out[1]), .in2(layer2_out[2]), .in3(layer2_out[3]), .in4(layer2_out[4]),
        .in5(layer2_out[5]), .in6(layer2_out[6]), .in7(layer2_out[7]), .in8(layer2_out[8]), .in9(layer2_out[9]),
        .max_index(predicted_digit)
    );

    // Control FSM (simple sequential flow)
    reg [1:0] state;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state <= 0;
            layer1_start <= 0;
            layer2_start <= 0;
        end else begin
            case (state)
                0: begin
                    if (start) begin
                        layer1_start <= 1;  // launch layer 1
                        state <= 1;
                    end
                end
                1: begin
                    layer1_start <= 0;
                    if (layer1_done) begin
                        layer2_start <= 1;  // launch layer 2
                        state <= 2;
                    end
                end
                2: begin
                    layer2_start <= 0;
                    if (layer2_done) begin
                        // Output is now valid: predicted_digit is set by argmax automatically
                        state <= 0;  // go back to idle or wait for next start
                    end
                end
            endcase
        end
    end
endmodule
