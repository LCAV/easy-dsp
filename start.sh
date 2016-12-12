#!/bin/sh
export LD_LIBRARY_PATH=/usr/local/lib

killall browser-main-daemon
killall browser-wsaudio
killall browser-wsconfig

nohup ./browser-main-daemon >> $1 2>&1 &
sleep 4
nohup ./browser-wsaudio >> $1 2>&1 &
nohup ./browser-wsconfig >> $1 2>&1 &
