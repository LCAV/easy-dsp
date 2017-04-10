#include "browser-wsaudio.h"


void* manage_clients(void* nothing);
void* send_config(libwebsock_client_state *state);
void* send_possible_config(libwebsock_client_state *state);

struct ws_client {
  libwebsock_client_state* c;
  struct ws_client* next;
};

struct ws_client* ws_clients = NULL;
struct ws_client* c = NULL;
pthread_mutex_t ws_client_lock;
unsigned int buffer_size;


void send_audio(char* buffer) {

  // Send the audio buffer to all clients
  struct ws_client* previous = NULL;
  pthread_mutex_lock(&ws_client_lock);
  for (c = ws_clients; c != NULL; c = c->next) {

    int re = libwebsock_send_binary(c->c, buffer, buffer_size);
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

void send_new_audio_config(void) {
  fprintf(stdout, "wsaudio: Sending new configuration buffer_frames=%d rate=%d channels=%d volume=%d.\n", buffer_frames, rate, channels, volume);

  buffer_size = (buffer_frames) * (channels) * EASY_DSP_AUDIO_FORMAT_BYTES;
  
  fprintf(stdout, "The buffer size is: %d\n", buffer_size);
     
  // Send the new configuration to clients
  for (c = ws_clients; c != NULL; c = c->next)
    send_config(c->c);
}


int onmessage_audio(libwebsock_client_state *state, libwebsock_message *msg) {
  fprintf(stderr, "Received message from client: %d\n", state->sockfd);
  fprintf(stderr, "Message opcode: %d\n", msg->opcode);
  fprintf(stderr, "Payload Length: %llu\n", msg->payload_len);
  fprintf(stderr, "Payload: %s\n", msg->payload);
  //now let's send it back.
  libwebsock_send_text(state, msg->payload);
  return 0;
}

int onopen_audio(libwebsock_client_state *state) {
  // printf("open\n");
  send_config(state);
  send_possible_config(state);
  struct ws_client* new_client = malloc(sizeof(struct ws_client));
  new_client->next = ws_clients;
  new_client->c = state;
  ws_clients = new_client;
  fprintf(stderr, "onopen: %d\n", state->sockfd);
  return 0;
}

void* send_config(libwebsock_client_state *state)
{
  char conf[100];
  char* c = conf;
  sprintf(conf, "{\"buffer_frames\":%d,\"rate\":%d,\"channels\":%d,\"volume\":%d}", buffer_frames, rate, channels, volume); 
  libwebsock_send_text(state, c);

  return NULL;
}

void* send_possible_config(libwebsock_client_state *state)
{
  char conf[100];
  char* c = conf;

  // place channel info into string
  char s_channel[20] = {0};
  int n = 0;
  for (unsigned int i = 0; i < numChannelConfigs; i++) {
    if (i == 0)
        n += sprintf (&s_channel[n], "[%u,", channelConfigs[i]);
    else if (i == numChannelConfigs - 1)
        n += sprintf (&s_channel[n], "%u]", channelConfigs[i]);
    else
        n += sprintf (&s_channel[n], "%u,", channelConfigs[i]);
  } 
  printf ("\nPossible number of channels: %s\n", s_channel);

  // place rate info into string
  char s_rates[20] = {0};
  n = 0;
  for (unsigned int i = 0; i < numPossibleRates; i++) {
    if (i == 0)
        n += sprintf (&s_rates[n], "[%u,", possibleRates[i]);
    else if (i == numPossibleRates - 1)
        n += sprintf (&s_rates[n], "%u]", possibleRates[i]);
    else
        n += sprintf (&s_rates[n], "%u,", possibleRates[i]);
  } 
  printf ("\nPossible rates: %s\n", s_rates);

  // format as JSON
  sprintf(conf, "{\"possible_channel\":%s,\"possible_rates\":%s}", 
      s_channel, s_rates); 
  printf("\n%s\n", conf);
  libwebsock_send_text(state, c);

  return NULL;
}

int onclose_audio(libwebsock_client_state *state)
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

void wsaudio_main(void) {

  buffer_size = (buffer_frames) * (channels) * EASY_DSP_AUDIO_FORMAT_BYTES;

  libwebsock_context *ctx = NULL;
  ctx = libwebsock_init();
  if(ctx == NULL) {
    fprintf(stderr, "Error during libwebsock_init.\n");
    exit(1);
  }
  libwebsock_bind(ctx, "0.0.0.0", "7321");
  fprintf(stdout, "libwebsock listening on port 7321\n");
  ctx->onmessage = onmessage_audio;
  ctx->onopen = onopen_audio;
  ctx->onclose = onclose_audio;
  libwebsock_wait(ctx);

  return NULL;
}

