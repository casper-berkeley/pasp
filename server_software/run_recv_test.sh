#!/bin/sh

platform=$1

taskset 04 nice -n -20 software/pasp_recv >out/recv_out 2>out/recv_err &
taskset 08 nice -n -20 software/pasp_record >out/record_out 2>out/record_err &
#nice -n -20 software/pasp_recv >out/recv_out 2>out/recv_err &
#nice -n -20 software/pasp_record >out/record_out 2>out/record_err &


echo configure_${platform}
cd configure_${platform}
./pasp_init.py

cd ..

killall -INT software/pasp_record
killall -INT software/pasp_recv

echo "regread pasp/dist_gbe/packet_count" | nc -u -w 1 192.168.0.4 7
