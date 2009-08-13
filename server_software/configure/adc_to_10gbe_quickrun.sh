#!/bin/sh

IP=192.168.1.2
PORT=7

cat adc_to_10gbe_init.txt | nc -u -w 1 $IP $PORT
cat adc_to_10gbe_ip_init.txt | nc -u -w 1 $IP $PORT
sleep 1m
cat adc_to_10gbe_ip_clear.txt | nc -u -w 1 $IP $PORT
#sleep 2m
