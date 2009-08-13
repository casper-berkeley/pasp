#!/bin/sh

IP=$ADC_TO_10GBE_IP
PORT=7
OUT_DIR=out

echo "regwrite reg_adcscope_trigger 1" | nc -u -q 1 $IP $PORT
sleep 1
echo "regwrite reg_adcscope_trigger 0" | nc -u -q 1 $IP $PORT
echo "bramdump bram_adcscope_1" | nc -u -q 7 $IP $PORT > $OUT_DIR/bram_adcscope_1
#./bram2dec.sh $OUT_DIR/bram_adcscope_1
echo "bramdump bram_adcscope_2" | nc -u -q 7 $IP $PORT > $OUT_DIR/bram_adcscope_2

