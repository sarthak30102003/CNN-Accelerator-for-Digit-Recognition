// argmax.v - Find index of maximum value among 10 signed inputs
module argmax(
    input  signed [31:0] in0, in1, in2, in3, in4, in5, in6, in7, in8, in9,
    output reg [3:0] max_index
);
    reg signed [31:0] best;
    integer idx;
    always @(*) begin
        best = in0; idx = 0;
        if (in1 > best) begin best = in1; idx = 1; end
        if (in2 > best) begin best = in2; idx = 2; end
        if (in3 > best) begin best = in3; idx = 3; end
        if (in4 > best) begin best = in4; idx = 4; end
        if (in5 > best) begin best = in5; idx = 5; end
        if (in6 > best) begin best = in6; idx = 6; end
        if (in7 > best) begin best = in7; idx = 7; end
        if (in8 > best) begin best = in8; idx = 8; end
        if (in9 > best) begin best = in9; idx = 9; end
        max_index = idx;
    end
endmodule
