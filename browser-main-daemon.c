/*
  The Main Daemon
  gcc -o browser-main-daemon -lasound browser-main-daemon.c
  ./browser-main-daemon
*/

#include <stdio.h>
#include <stdlib.h>
#include <alsa/asoundlib.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <pthread.h>
#include <signal.h>

void sig_handler(int signo)
{
  if (signo == SIGPIPE) {
    fprintf(stdout, "received SIGPIPE\n");
  }
}

struct client {
  int addr;
  struct client* next;
};

void* handle_connections_audio(void* nothing);
void* handle_connections_control(void* nothing);
void* handle_audio(void* nothing);

struct client* clients;
pthread_t* audio_thread;

// alsa parameters
snd_pcm_t *capture_handle;
int* buffer_frames;
unsigned int* rate;
int* volume;
int* channels;


main (int argc, char *argv[])
{
  buffer_frames = malloc(sizeof(*buffer_frames));
  rate = malloc(sizeof(*rate));
  channels = malloc(sizeof(*channels));
  volume = malloc(sizeof(*volume));
  audio_thread = malloc(sizeof(*audio_thread));
  *buffer_frames = 4096;
  *rate = 48000;
  *channels = 6;
  *volume = 80;
  clients = NULL;

  // "catch" SIGPIPE we get when we try to send data to a disconnected client
  if (signal(SIGPIPE, sig_handler) == SIG_ERR) {
    printf("\ncan't catch SIGPIPE\n");
  }

  pthread_t connections_audio_thread;
  pthread_t connections_control_thread;


  if( pthread_create(&connections_audio_thread, NULL, handle_connections_audio, NULL) < 0) {
      perror("could not create thread to handle connections audio");
      return 1;
  }
  if( pthread_create(audio_thread, NULL, handle_audio, NULL) < 0) {
      perror("could not create thread to handle audio ALSA");
      return 1;
  }
  if( pthread_create(&connections_control_thread, NULL, handle_connections_control, NULL) < 0) {
      perror("could not create thread to handle connections control");
      return 1;
  }

  while (1) {
  }

  exit (0);
}

void* handle_audio(void* nothing) {

  int i, j;
  int err;
  char *buffer;
  // int buffer_frames = 176400/10;
  int buffer_size;
  // unsigned int rate = 176400;
  snd_pcm_hw_params_t *hw_params;
	snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

  if ((err = snd_pcm_open (&capture_handle, "hw:0", SND_PCM_STREAM_CAPTURE, 0)) < 0) {
    fprintf (stderr, "cannot open audio device %s (%s)\n",
             "hw:0",
             snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "audio interface opened\n");

  if ((err = snd_pcm_hw_params_malloc (&hw_params)) < 0) {
    fprintf (stderr, "cannot allocate hardware parameter structure (%s)\n",
             snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params allocated\n");

  if ((err = snd_pcm_hw_params_any (capture_handle, hw_params)) < 0) {
    fprintf (stderr, "cannot initialize hardware parameter structure (%s)\n",
             snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params initialized\n");

  if ((err = snd_pcm_hw_params_set_access (capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
    fprintf (stderr, "cannot set access type (%s)\n",
             snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params access set\n");

  if ((err = snd_pcm_hw_params_set_format (capture_handle, hw_params, format)) < 0) {
    fprintf (stderr, "cannot set sample format (%s)\n",
             snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params format set\n");

  if ((err = snd_pcm_hw_params_set_rate_near (capture_handle, hw_params, rate, 0)) < 0) {
    fprintf (stderr, "cannot set sample rate (%s)\n",
             snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params rate set\n");

  if ((err = snd_pcm_hw_params_set_channels (capture_handle, hw_params, *channels)) < 0) {
    fprintf (stderr, "cannot set channel count (%s)\n",
             snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params channels set\n");

  if ((err = snd_pcm_hw_params (capture_handle, hw_params)) < 0) {
    fprintf (stderr, "cannot set parameters (%s)\n",
             snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params set\n");

  snd_pcm_hw_params_free (hw_params);

  fprintf(stdout, "hw_params freed\n");

  if ((err = snd_pcm_prepare (capture_handle)) < 0) {
    fprintf (stderr, "cannot prepare audio interface for use (%s)\n",
             snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "audio interface prepared\n");

  buffer_size = *buffer_frames * snd_pcm_format_width(format) / 8 * (*channels);
  if ((buffer = (char*) malloc(buffer_size)) == NULL) {
    fprintf(stdout, "Cannot allocate buffer %p (size: %d)\n", buffer, buffer_size);
    exit(1);
  }

  fprintf(stdout, "buffer allocated %d\n", buffer_size);

  long min, max;
  snd_mixer_t *handle;
  int iii;
  snd_mixer_selem_id_t *sid;
  const char *card = "default";
  char *selem_name = malloc(3* sizeof(char));
  snd_mixer_open(&handle, 0);
  snd_mixer_attach(handle, card);
  snd_mixer_selem_register(handle, NULL, NULL);
  snd_mixer_load(handle);
  snd_mixer_selem_id_alloca(&sid);
  snd_mixer_selem_id_set_index(sid, 0);
  for (iii = 1; iii <= 3; iii++) {
    sprintf(selem_name, "Ch%d", iii);
    snd_mixer_selem_id_set_name(sid, selem_name);
    snd_mixer_elem_t* elem = snd_mixer_find_selem(handle, sid);
    snd_mixer_selem_get_capture_volume_range(elem, &min, &max);
    printf("Min max %d %d\n", min, max);
    int ee = snd_mixer_selem_set_capture_volume_all(elem, (*volume) * (max - min) / 100);
    printf("Error: %d\n", ee);
    long vv;
    snd_mixer_selem_get_capture_volume(elem, 1, &vv);
    printf("New volume: %d\n", vv);
    // snd_mixer_close(handle);
  }

  for (i = 0; i < 10000000; ++i) {
    fprintf(stdout, "000 %d %d\n", i, *buffer_frames);
    if ((err = snd_pcm_readi (capture_handle, buffer, *buffer_frames)) != *buffer_frames) {
      fprintf (stderr, "read from audio interface failed (%s)\n",
               err, snd_strerror (err));
      exit (1);
    }
    // fprintf(stdout, "001\n");

    // printf("Received %d %d %d %d\n", buffer[0], buffer[1], buffer[2], buffer[3]);
    short* t = buffer;
    // printf("Received %d %d %d\n", buffer[0], buffer[1], t[0]);
    // fprintf(stdout, "read %d done\n", i);
    struct client* c = clients;
    struct client* previous = NULL;
    while (c != NULL) {
      fprintf(stdout, "Send to client %d %d %d %d\n", buffer[0], buffer[1], buffer[2], buffer[3]);
      int re = write((*c).addr, buffer, buffer_size);
      if (re == -1) {
        // This client is gone
        // We remove it
        if (previous == NULL) { // First client
          clients = (*c).next;
        } else {
          (*previous).next = (*c).next;
        }
      }
      previous = c;
      c = (*c).next;
    }
  }

  free(buffer);

  fprintf(stdout, "buffer freed\n");

  snd_pcm_close (capture_handle);
  fprintf(stdout, "audio interface closed\n");
}

void* handle_connections_control(void* nothing) {
  const char *SOCKNAMEC = "/tmp/micros-control.socket";
  unlink(SOCKNAMEC);
  int sfd, t, s2;
  struct sockaddr_un addr, remote;
  int config[4];
  int* c = config;

  sfd = socket(AF_UNIX, SOCK_STREAM, 0);            /* Create socket */
  if (sfd == -1) {
    fprintf (stderr, "cannot create the socket control\n");
    return;
  }

  memset(&addr, 0, sizeof(struct sockaddr_un));     /* Clear structure */
  addr.sun_family = AF_UNIX;                            /* UNIX domain address */
  strncpy(addr.sun_path, SOCKNAMEC, sizeof(addr.sun_path) - 1);

  if (bind(sfd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un)) == -1) {
    fprintf (stderr, "cannot bind the socket control\n");
    return;
  }

  listen(sfd, 3);
  fprintf (stdout, "Bind successful control\n");
  while (1) {
    fprintf(stdout, "Waiting for a connection...\n");
    t = sizeof(remote);
    if ((s2 = accept(sfd, (struct sockaddr *)&remote, &t)) == -1) {
      fprintf (stderr, "cannot accept the connection control\n");
      continue;
    }

    fprintf(stdout, "New client control\n");

    recv(s2, c, sizeof(config), 0);
    fprintf(stdout, "111\n");
    pthread_cancel(*audio_thread);
    fprintf(stdout, "222\n");
    sleep(2);
    snd_pcm_close (capture_handle);
    fprintf(stdout, "333\n");
    sleep(1);

    *buffer_frames = config[0];
    *rate = config[1];
    *channels = config[2];
    *volume = config[3];

    struct client* client;
    for (client = clients; client != NULL; client = (*client).next) {
      write((*client).addr, c, sizeof(config));
    }

    fprintf(stdout, "New configuration: %d %d %d %d", *buffer_frames, *rate, *channels, *volume);

    if( pthread_create(audio_thread, NULL, handle_audio, NULL) < 0) {
        perror("could not create thread to handle audio ALSA");
        return;
    }

  }
}

void* handle_connections_audio(void* nothing) {
  const char *SOCKNAME = "/tmp/micros-audio.socket";
  unlink(SOCKNAME);
  int sfd, t, s2;
  struct sockaddr_un addr, remote;
  int config[4];
  int* c = config;

  sfd = socket(AF_UNIX, SOCK_STREAM, 0);            /* Create socket */
  if (sfd == -1) {
    fprintf (stderr, "cannot create the socket audio\n");
    return;
  }

  memset(&addr, 0, sizeof(struct sockaddr_un));     /* Clear structure */
  addr.sun_family = AF_UNIX;                            /* UNIX domain address */
  strncpy(addr.sun_path, SOCKNAME, sizeof(addr.sun_path) - 1);

  if (bind(sfd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un)) == -1) {
    fprintf (stderr, "cannot bind the socket\n");
    return;
  }

  listen(sfd, 3);
  fprintf (stdout, "Bind successful audio\n");
  while (1) {
    fprintf(stdout, "Waiting for a connection...\n");
    t = sizeof(remote);
    if ((s2 = accept(sfd, (struct sockaddr *)&remote, &t)) == -1) {
      fprintf (stderr, "cannot accept the connection audio\n");
      continue;
    }

    fprintf(stdout, "New client audio\n");

    config[0] = *buffer_frames;
    config[1] = *rate;
    config[2] = *channels;
    config[3] = *volume;
    write(s2, c, sizeof(config));

    struct client* new_client = malloc(sizeof(struct client));
    (*new_client).addr = s2;
    (*new_client).next = clients;
    clients = new_client;
  }
}
