#!/bin/bash

if [ "${#}" -eq 0 ]; then
    echo 'Error: ${1} must be the name of a log file.' 2>&1
    exit 1
fi

log_file="${1}"

./stop.sh > /dev/null

echo -n 'Starting daemons '
nohup ./browser-main-daemon >> "${log_file}" 2>&1 &
sleep 4; echo -n '.'
nohup ./browser-wsaudio     >> "${log_file}" 2>&1 &
sleep 1; echo -n '.'
nohup ./browser-wsconfig    >> "${log_file}" 2>&1 &
sleep 1; echo -n '.'
echo " done"
