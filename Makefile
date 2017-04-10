CC = gcc
CFLAGS = -g -Wall -Wextra -O0 -std=gnu99

default: all

all: browser-main-daemon

browser-main-daemon: browser-main-daemon.c
	$(CC) $(CFLAGS) browser-main-daemon.c browser-wsconfig.c browser-wsaudio.c  -o browser-main-daemon -lasound -lpthread -lwebsock -ljansson

clean:
	-rm -f browser-main-daemon
