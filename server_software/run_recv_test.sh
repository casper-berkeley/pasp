#!/bin/sh

platform=$1

taskset 04 nice -n -20 software/pasp_recv >out/recv_out 2>out/recv_err &
taskset 08 nice -n -20 software/pasp_process >out/process_out 2>out/process_err &
#nice -n -20 software/pasp_recv >out/recv_out 2>out/recv_err &
#nice -n -20 software/pasp_process >out/process_out 2>out/process_err &


echo configure_${platform}
cd configure_${platform}
./pasp_init.py

cd ..

echo "sleep 600 to wait for threads to finish"
sleep 600

killall -INT software/pasp_process
killall -INT software/pasp_recv

echo "regread pasp/dist_gbe/packet_count" | nc -u -w 1 192.168.0.4 7
