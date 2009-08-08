#!/bin/sh

echo "regwrite pasp/dist_gbe/packet_count_rst 1" | nc -u -w 1 192.168.0.4 7
echo "regwrite pasp/dist_gbe/packet_count_rst 0" | nc -u -w 1 192.168.0.4 7
echo "regread pasp/dist_gbe/packet_count" | nc -u -w 1 192.168.0.4 7

taskset 04 nice -n -20 software/pasp_recv >out/recv_out 2>out/recv_err &
taskset 08 nice -n -20 software/pasp_record >out/record_out 2>out/record_err &
#nice -n -20 software/pasp_recv >out/recv_out 2>out/recv_err &
#nice -n -20 software/pasp_record >out/record_out 2>out/record_err &
cd configure
./quickrun.sh >out/quickrun_out 2>out/quickrun_err
cd ..

killall -INT software/pasp_record
killall -INT software/pasp_recv

echo "regread pasp/dist_gbe/packet_count" | nc -u -w 1 192.168.0.4 7
