#!/bin/bash

killall browser-wsconfig
killall browser-wsaudio
killall browser-main-daemon
rm logs.txt

echo "Daemons stopped"
