// relu.v - ReLU activation (zero if negative)
module relu(
    input  signed [31:0] din,   // 32-bit Q8.24 input
    output signed [31:0] dout   // 32-bit Q8.24 output
);
    // If sign bit is 1 (negative), output 0; else pass through
    assign dout = (din[31] == 1'b0) ? din : 32'b0;
endmodule
