#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <stdlib.h>
#include <websock/websock.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>

void sig_handler(int signo)
{
  if (signo == SIGPIPE) {
    fprintf(stdout, "received SIGPIPE\n");
  }
}

void* main_ws(void* nothing);
int* config;
void* send_config(libwebsock_client_state *state);

struct ws_client {
  libwebsock_client_state* c;
  struct ws_client* next;
};

struct ws_client* ws_clients;
pthread_mutex_t ws_client_lock;

int main(void)
{
  int s, t, len;
  struct sockaddr_un remote;
  char *buffer;
  char *buffer_temp;
  int buffer_pos = 0;
  int buffer_frames;
  const char *SOCKNAME = "/tmp/micros-audio.socket";
  ws_clients = NULL;
  int config_size = sizeof(int) * 4;
  config = malloc(config_size);

  if (signal(SIGPIPE, sig_handler) == SIG_ERR) {
    printf("\ncan't catch SIGPIPE\n");
  }

  pthread_t main_ws_thread;

  if( pthread_create(&main_ws_thread, NULL, main_ws, NULL) < 0) {
      perror("could not create thread for main websocket loop");
      return 1;
  }

  if ((s = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
      perror("socket");
      exit(1);
  }

  printf("Trying to connect...\n");

  remote.sun_family = AF_UNIX;
  strcpy(remote.sun_path, SOCKNAME);
  len = strlen(remote.sun_path) + sizeof(remote.sun_family);
  if (connect(s, (struct sockaddr *)&remote, len) == -1) {
      perror("connect");
      while(1) {

      }
      exit(1);
  }

  printf("Connected.\n");

  // First message is the config
  recv(s, config, config_size, 0);
  buffer_frames = config[0]*config[2]*2;

  buffer = malloc(buffer_frames);
  buffer_temp = malloc(buffer_frames);
  struct ws_client* c;

  while(1) {
    // printf("ok\n");
    if ((t=recv(s, buffer_temp, buffer_frames, 0)) > 0) {
      if (t == config_size) {
        // printf("New config\n");
        int* temp = (int*) buffer_temp;
        config[0] = temp[0];
        config[1] = temp[1];
        config[2] = temp[2];
        config[3] = temp[3];
        buffer_frames = config[0]*config[2]*2;
        // printf("New config %d %d %d %d\n", config[0], config[1], config[2], config[3]);
        free(buffer);
        free(buffer_temp);
        buffer = malloc(buffer_frames);
        buffer_temp = malloc(buffer_frames);
        buffer_pos = 0;
        for (c = ws_clients; c != NULL; c = c->next) {
          send_config(c->c);
        }
        continue;
      }
      // printf("%d %d %d", buffer_pos, t, buffer_frames);
      // We must fill completly buffer with buffer_frames length before sending it
      memcpy((buffer + buffer_pos), buffer_temp, t);
      if ((buffer_pos + t) < buffer_frames) {
        buffer_pos += t;
        continue;
      }
      buffer_pos = 0;
      // printf("ok2\n");
      struct ws_client* previous = NULL;
      pthread_mutex_lock(&ws_client_lock);
      for (c = ws_clients; c != NULL; c = c->next) {
        // printf("ici\n");
        int re = libwebsock_send_binary(c->c, buffer, buffer_frames);
        if (re == -1) {
          if (previous == NULL) {
            ws_clients = c->next;
          } else {
            previous->next = c->next;
          }
        } else {
          previous = c;
        }
      }
      pthread_mutex_unlock(&ws_client_lock);
      // printf("%d bytes received %d %d\n", t, buffer[0], buffer[1]);
    } else {
      if (t < 0) perror("recv");
      else printf("Server closed connection\n");
      break;
    }
  }

  close(s);

  return 0;
}

int
onmessage(libwebsock_client_state *state, libwebsock_message *msg)
{
  fprintf(stderr, "Received message from client: %d\n", state->sockfd);
  fprintf(stderr, "Message opcode: %d\n", msg->opcode);
  fprintf(stderr, "Payload Length: %llu\n", msg->payload_len);
  fprintf(stderr, "Payload: %s\n", msg->payload);
  //now let's send it back.
  libwebsock_send_text(state, msg->payload);
  return 0;
}

int
onopen(libwebsock_client_state *state)
{
  // printf("open\n");
  send_config(state);
  struct ws_client* new_client = malloc(sizeof(struct ws_client));
  new_client->next = ws_clients;
  new_client->c = state;
  ws_clients = new_client;
  fprintf(stderr, "onopen: %d\n", state->sockfd);
  return 0;
}

void*
send_config(libwebsock_client_state *state)
{
  char conf[100];
  char* c = conf;
  sprintf(conf, "{\"buffer_frames\":%d,\"rate\":%d,\"channels\":%d,\"volume\":%d}", config[0], config[1], config[2], config[3]);
  libwebsock_send_text(state, c);

  return NULL;
}

int
onclose(libwebsock_client_state *state)
{
  pthread_mutex_lock(&ws_client_lock);
  struct ws_client* c;
  struct ws_client* previous = NULL;
  for (c = ws_clients; c != NULL; c = c->next) {
    if (c->c == state) {
      break;
    }
    previous = c;
  }
  if (previous == NULL) {
    ws_clients = c->next;
  } else {
    previous->next = c->next;
  }
  pthread_mutex_unlock(&ws_client_lock);
  fprintf(stderr, "onclose: %d\n", state->sockfd);
  return 0;
}

void* main_ws(void* nothing) {
  libwebsock_context *ctx = NULL;
  ctx = libwebsock_init();
  if(ctx == NULL) {
    fprintf(stderr, "Error during libwebsock_init.\n");
    exit(1);
  }
  libwebsock_bind(ctx, "0.0.0.0", "7321");
  fprintf(stdout, "libwebsock listening on port 7321\n");
  ctx->onmessage = onmessage;
  ctx->onopen = onopen;
  ctx->onclose = onclose;
  libwebsock_wait(ctx);

  return NULL;
}
