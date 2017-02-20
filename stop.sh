#!/bin/bash

killall browser-main-daemon > /dev/null 2>&1
killall browser-wsaudio     > /dev/null 2>&1
killall browser-wsconfig    > /dev/null 2>&1

echo "Daemons stopped"
