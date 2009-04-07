#!/bin/sh

IP=192.168.0.4
PORT=7

cat pasp_init.txt | nc -u -w 1 $IP $PORT
cat ip_init.txt | nc -u -w 1 $IP $PORT
sleep 5m
cat ip_clear.txt | nc -u -w 1 $IP $PORT
#sleep 2m
