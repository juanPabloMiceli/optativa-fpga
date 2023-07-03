# $1 bitstream
# $2 archivo de entrada FPGA

dir=$(pwd)
source xilinx_env.sh
convert_to_xsvf nexys2-500 $1 $dir/top_level.xsvf
cd $dir
./flLoop.py -b nexys2-500 $2 -x top_level.xsvf
exec bash
