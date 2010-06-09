#!/bin/sh

platform=roach
software_dir=software
fftlib=japan

taskset 02 nice -n -20 ${software_dir}/pasp_recv >out/recv_out 2>out/recv_err &
taskset 04 nice -n -20 ${software_dir}/pasp_distribute >out/distribute_out 2>out/distribute_err &
taskset 08 nice -n -20 ${software_dir}/${fftlib}_pasp_process >out/process_out 2>out/process_err &
#nice -n -20 ${software_dir}/pasp_recv >out/recv_out 2>out/recv_err &
#nice -n -20 ${software_dir}/pasp_process >out/process_out 2>out/process_err &


echo configure_${platform}
cd configure_${platform}
./pasp_main.py

cd ..

echo "sleep 60 to wait for threads to finish"
sleep 60

killall -INT ${software_dir}/${fftlib}_pasp_process
killall -INT ${software_dir}/pasp_distribute
killall -INT ${software_dir}/pasp_recv

echo "regread pasp/dist_gbe/packet_count" | nc -u -w 1 192.168.0.4 7
