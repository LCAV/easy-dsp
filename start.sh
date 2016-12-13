#!/bin/sh
export LD_LIBRARY_PATH=/usr/local/lib

killall browser-main-daemon >> /dev/null 2>&1
killall browser-wsaudio >> /dev/null 2>&1
killall browser-wsconfig >> /dev/null 2>&1

nohup ./browser-main-daemon >> $1 2>&1 &
sleep 4
nohup ./browser-wsaudio >> $1 2>&1 &
nohup ./browser-wsconfig >> $1 2>&1 &

print "Daemons started!"
