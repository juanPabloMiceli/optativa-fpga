ghdl -a CordicIter.vhd Cordic.vhd Cordic_tb.vhd 
ghdl -e cordic_tb
ghdl -r cordic_tb --vcd=out.vcd
