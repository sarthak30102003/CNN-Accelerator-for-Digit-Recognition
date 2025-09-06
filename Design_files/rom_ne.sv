`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 22.08.2025 22:08:01
// Design Name: 
// Module Name: rom_ne
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
// rom_neuron.sv
module rom_neuron #(
    parameter DEPTH = 784,
    parameter WIDTH = 32,
    parameter ID = 0
) (
    input  wire clk,
    input  wire [$clog2(DEPTH)-1:0] addr,
    output reg  [WIDTH-1:0] dout
);
    reg [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        string fname;
        $sformat(fname, "W1_neuron_%0d.mem", ID);
        $readmemh(fname, mem);
    end

    // synchronous read (1-cycle latency)
    always @(posedge clk) begin
        dout <= mem[addr];
    end
endmodule
