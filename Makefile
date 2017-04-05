CC = gcc
CFLAGS = -g -Wall -Wextra -O0 -std=gnu99

default: all

all: browser-main-daemon browser-wsaudio

browser-main-daemon: browser-main-daemon.c
	$(CC) $(CFLAGS) browser-main-daemon.c browser-wsconfig.c -o browser-main-daemon -lasound -lpthread -lwebsock -ljansson

browser-wsaudio: browser-wsaudio.c
	$(CC) $(CFLAGS) browser-wsaudio.c     -o browser-wsaudio     -lwebsock   -lpthread

clean:
	-rm -f browser-main-daemon
	-rm -f browser-wsaudio
