default: program

program:
	gcc -o browser-main-daemon -lasound browser-main-daemon.c
	LD_LIBRARY_PATH=/usr/local/lib gcc -g -O2 -o browser-wsconfig browser-wsconfig.c -lwebsock -ljansson
	gcc -g -O2 -o browser-wsaudio browser-wsaudio.c -lwebsock

clean:
	-rm -f browser-main-daemon
	-rm -f browser-wsconfig
	-rm -f browser-wsaudio
