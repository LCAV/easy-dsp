#include <sys/socket.h>
#include <sys/un.h>
#include <stdio.h>
#include <stdlib.h>
#include <websock/websock.h>
#include <pthread.h>
#include <string.h>
#include <jansson.h>
#include <unistd.h>

int
onmessage(libwebsock_client_state *state, libwebsock_message *msg)
{
  fprintf(stderr, "Received message from client: %d\n", state->sockfd);
  fprintf(stderr, "Message opcode: %d\n", msg->opcode);
  fprintf(stderr, "Payload Length: %llu\n", msg->payload_len);
  fprintf(stderr, "Payload: %s\n", msg->payload);
  json_t *root, *channels, *rate, *buffer_frames, *volume;
  json_error_t error;
  root = json_loads(msg->payload, 0, &error);
  if(!root) {
    fprintf(stderr, "error: on line %d: %s\n", error.line, error.text);
    return 0;
  }
  channels = json_object_get(root, "channels");
  rate = json_object_get(root, "rate");
  buffer_frames = json_object_get(root, "buffer_frames");
  volume = json_object_get(root, "volume");
  int config[4];
  int* c = config;

  config[0] = json_integer_value(buffer_frames);
  config[1] = json_integer_value(rate);
  config[2] = json_integer_value(channels);
  config[3] = json_integer_value(volume);

  printf("New config: %d %d %d %d\n", config[0], config[1], config[2], config[3]);

  int s, len;
  struct sockaddr_un remote;
  const char *SOCKNAME = "/tmp/micros-control.socket";

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
      exit(1);
  }

  printf("Connected.\n");

  write(s, c, sizeof(config));
  printf("Config sent\n");

  close(s);

  return 0;
}

int
onopen(libwebsock_client_state *state)
{
  fprintf(stderr, "onopen: %d\n", state->sockfd);
  return 0;
}

int
onclose(libwebsock_client_state *state)
{
  fprintf(stderr, "onclose: %d\n", state->sockfd);
  return 0;
}

int main(void)
{
  libwebsock_context *ctx = NULL;
  ctx = libwebsock_init();
  if(ctx == NULL) {
    fprintf(stderr, "Error during libwebsock_init.\n");
    exit(1);
  }
  libwebsock_bind(ctx, "0.0.0.0", "7322");
  fprintf(stdout, "libwebsock listening on port 7322\n");
  ctx->onmessage = onmessage;
  ctx->onopen = onopen;
  ctx->onclose = onclose;
  libwebsock_wait(ctx);
  return 0;
}
