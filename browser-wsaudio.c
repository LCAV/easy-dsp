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

void sig_handler(int signo)
{
  if (signo == SIGPIPE) {
    fprintf(stdout, "received SIGPIPE\n");
  }
}

void* main_ws(void* nothing);
config_t audio_cfg;
void* send_config(libwebsock_client_state *state);

struct ws_client {
  libwebsock_client_state* c;
  struct ws_client* next;
};

struct ws_client* ws_clients;
pthread_mutex_t ws_client_lock;

int main(void)
{
  int s, len;
  ssize_t t;
  struct sockaddr_un remote;
  char *buffer = NULL;
  char *buffer_temp = NULL;
  int buffer_frames = 0;
  const char *SOCKNAME = EASY_DSP_AUDIO_SOCKET;
  ws_clients = NULL;

  easy_dsp_hdr_t magic_byte;  // magic number for audio/config
  config_t audio_cfg_local;
  audio_cfg_local.config.buffer_frames = 0;

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

  struct ws_client* c;

  int start_flag = 0;

  while(true) {

    // First we get the magic byte to know if we expect an audio or config packet
    t = recv(s, &magic_byte, sizeof(easy_dsp_hdr_t), MSG_WAITALL);
    if (t != sizeof(easy_dsp_hdr_t))
        goto close_connection;

    if (start_flag == 0)
    {
      printf("%X %d\n", magic_byte, t);
      start_flag = 1;
    }

    if (magic_byte == EASY_DSP_HDR_CONFIG)
    {
      // Get a new configuration
      t = recv(s, &audio_cfg_local, sizeof(config_t), MSG_WAITALL);
      if (t != sizeof(config_t))
          goto close_connection;

      // copy to global config
      audio_cfg = audio_cfg_local;

      // new audio buffer size in bytes
      int new_buffer_frames = audio_cfg.config.buffer_frames * audio_cfg.config.channels * EASY_DSP_AUDIO_FORMAT_BYTES;
      
      // if the buffer size has changed, update the malloc
      if (new_buffer_frames != buffer_frames)
      {

        if (buffer != NULL)
          free(buffer);

        if (buffer_temp != NULL)
          free(buffer_temp);

        buffer_frames = new_buffer_frames;

        buffer = (char *)malloc(buffer_frames * sizeof(char));
        buffer_temp = (char *)malloc(buffer_frames * sizeof(char));


      }

      fprintf(stdout, "wsaudio: Sending new configuration buffer_frames=%d rate=%d channels=%d volume=%d.\n",
          audio_cfg.config.buffer_frames, audio_cfg.config.rate, audio_cfg.config.channels, audio_cfg.config.volume);
      fprintf(stdout, "The buffer size is: %d\n", buffer_frames);
     
      // Send the new configuration to clients
      for (c = ws_clients; c != NULL; c = c->next)
        send_config(c->c);

    }
    else if (magic_byte == EASY_DSP_HDR_AUDIO)
    {
      // We should receive the new buffer of audio data
      t = recv(s, buffer, buffer_frames, MSG_WAITALL);
      if (t != buffer_frames)
        goto close_connection;

      // swap the buffers
      memcpy(buffer_temp, buffer, buffer_frames);
      char *tmp = buffer;
      buffer  = buffer_temp;
      buffer_temp = tmp;
      
      // Send the audio buffer to all clients
      struct ws_client* previous = NULL;
      pthread_mutex_lock(&ws_client_lock);
      for (c = ws_clients; c != NULL; c = c->next) {

        int re = libwebsock_send_binary(c->c, buffer_temp, buffer_frames);
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

    }

  }

close_connection:
  // Free allocated buffers
  if (buffer != NULL)
    free(buffer);
  if (buffer_temp != NULL)
    free(buffer_temp);

  printf("Closed connection.\n");

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
  sprintf(conf, "{\"buffer_frames\":%d,\"rate\":%d,\"channels\":%d,\"volume\":%d}", 
      audio_cfg.config.buffer_frames, audio_cfg.config.rate, audio_cfg.config.channels, audio_cfg.config.volume); 
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
