#ifndef _BROWSER_WSAUDIO_H_
#define _BROWSER_WSAUDIO_H_

#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <stdlib.h>
#include <websock/websock.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdint.h>

#include "browser-config.h"
#include "global.h"

void wsaudio_main(void);
void send_audio(char*);
void send_new_audio_config(void);


#endif /* _BROWSER_WSAUDIO_H_ */