#ifndef _BROWSER_WSCONFIG_H_
#define _BROWSER_WSCONFIG_H_

#include <sys/un.h>
#include <stdio.h>
#include <stdlib.h>
#include <websock/websock.h>
#include <pthread.h>
#include <string.h>
#include <jansson.h>
#include <unistd.h>

#include "browser-config.h"

void wsconfig_main(void);


#endif /* _BROWSER_WSCONFIG_H_ */