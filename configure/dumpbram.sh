#!/bin/sh

IP=192.168.0.4
PORT=7
OUT_DIR=out

echo "regwrite pasp/reg_adcscope_trigger 1" | nc -u -q 1 $IP $PORT
sleep 1
echo "regwrite pasp/reg_adcscope_trigger 0" | nc -u -q 1 $IP $PORT
echo "bramdump pasp/bram_adcscope_1" | nc -u -q 7 $IP $PORT > $OUT_DIR/bram_adcscope_1
#./bram2dec.sh $OUT_DIR/bram_adcscope_1
echo "bramdump pasp/bram_adcscope_2" | nc -u -q 7 $IP $PORT > $OUT_DIR/bram_adcscope_2
#./bram2dec.sh $OUT_DIR/bram_adcscope_2
echo "bramdump pasp/scope_output1/bram" | nc -u -w 2 $IP $PORT >$OUT_DIR/scope1
echo "bramdump pasp/scope_output2/bram" | nc -u -w 2 $IP $PORT >$OUT_DIR/scope2
echo "bramdump pasp/scope_output3/bram" | nc -u -w 2 $IP $PORT >$OUT_DIR/scope3
echo "bramdump pasp/scope_output4/bram" | nc -u -w 2 $IP $PORT >$OUT_DIR/scope4
echo "bramdump pasp/scope_output5/bram" | nc -u -w 2 $IP $PORT >$OUT_DIR/scope5
echo "bramdump pasp/scope_output6/bram" | nc -u -w 2 $IP $PORT >$OUT_DIR/scope6
echo "bramdump pasp/scope_output7/bram" | nc -u -w 2 $IP $PORT >$OUT_DIR/scope7
echo "bramdump pasp/scope_output8/bram" | nc -u -w 2 $IP $PORT >$OUT_DIR/scope8
