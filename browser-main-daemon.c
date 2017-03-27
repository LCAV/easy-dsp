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
#include <stdbool.h>
#include <unistd.h>
#include <stdint.h>

#include "browser-config.h"

#define ARRAY_SIZE(a) (sizeof(a) / sizeof *(a))


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
pthread_mutex_t audio_client_lock;

// alsa parameters
snd_pcm_t *capture_handle;
int* buffer_frames;
unsigned int* rate;
int* volume;
int* channels;

// possible alsa configurations
unsigned int* channelConfigs;
unsigned int numChannelConfigs = 0;
unsigned int* possibleRates;
unsigned int numPossibleRates = 0;

// a few flags
int audio_thread_active_flag = 0;
int new_config_received_flag = 0;


int main (int argc, char *argv[])
{
  buffer_frames = malloc(sizeof(*buffer_frames));
  rate = malloc(sizeof(*rate));
  channels = malloc(sizeof(*channels));
  volume = malloc(sizeof(*volume));
  audio_thread = malloc(sizeof(*audio_thread));
  *buffer_frames = EASY_DSP_BUFFER_SIZE_BYTES;
  *rate = EASY_DSP_AUDIO_FREQ_HZ;
  *channels = EASY_DSP_NUM_CHANNELS;
  *volume = EASY_DSP_VOLUME;
  clients = NULL;

  query_available_config();

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
  else
    audio_thread_active_flag = 1;  // Let the world know we are active

  if( pthread_create(&connections_control_thread, NULL, handle_connections_control, NULL) < 0) {
      perror("could not create thread to handle connections control");
      return 1;
  }

    while (true) {
        // Sleep for a long time to not take CPU cycles. ANY constant could work
        // here.
        sleep(10);
    }

  exit (0);
}

void query_available_config(void)
{
  unsigned int i;
  int err;
  unsigned int minval, maxval;
  unsigned int numReadVals;
  static const unsigned int rates[] = {
    5512,
    8000,
    11025,
    16000,
    22050,
    32000,
    44100,
    48000,
    64000,
    88200,
    96000,
    176400,
    192000,
  };

  const char *device_name = "hw:0";
  snd_pcm_hw_params_t *hw_params;

  err = snd_pcm_open(&capture_handle, device_name, SND_PCM_STREAM_CAPTURE, 0);
  if (err < 0) {
      fprintf(stderr, "cannot open device '%s': %s\n", device_name, snd_strerror(err));
      return 1;
  }

  snd_pcm_hw_params_alloca(&hw_params);
  err = snd_pcm_hw_params_any(capture_handle, hw_params);
  if (err < 0) {
      fprintf(stderr, "cannot get hardware parameters: %s\n", snd_strerror(err));
      snd_pcm_close(capture_handle);
      return 1;
  }

  // query possible channel config
  numReadVals = 0;
  err = snd_pcm_hw_params_get_channels_min(hw_params, &minval);
  if (err < 0) {
      fprintf(stderr, "cannot get minimum channels count: %s\n", snd_strerror(err));
      snd_pcm_close(capture_handle);
      return 1;
  }
  err = snd_pcm_hw_params_get_channels_max(hw_params, &maxval);
  if (err < 0) {
      fprintf(stderr, "cannot get maximum channels count: %s\n", snd_strerror(err));
      snd_pcm_close(capture_handle);
      return 1;
  }
  for (i = minval; i <= maxval; ++i) {
      if (!snd_pcm_hw_params_test_channels(capture_handle, hw_params, i))
          numChannelConfigs++;
  }
  channelConfigs = malloc(sizeof(unsigned int)*numChannelConfigs);
  for (i = minval; i <= maxval; ++i) {
      if (!snd_pcm_hw_params_test_channels(capture_handle, hw_params, i)) {
          channelConfigs[numReadVals] = i;
          numReadVals++;
        }
  }
  printf("Possible number of channels:");
  for (i = 0; i < numChannelConfigs; i++) {
    printf(" %u", channelConfigs[i]);
  }
  putchar('\n');

  // query possible sample rate config
  numReadVals = 0;
  for (i = 0; i < ARRAY_SIZE(rates); ++i) {
    if (!snd_pcm_hw_params_test_rate(capture_handle, hw_params, rates[i], 0)){
      numPossibleRates++;
    }
  }
  possibleRates = malloc(sizeof(unsigned int)*numPossibleRates);
  for (i = 0; i < ARRAY_SIZE(rates); ++i) {
    if (!snd_pcm_hw_params_test_rate(capture_handle, hw_params, rates[i], 0)){
      possibleRates[numReadVals] = rates[i];
      numReadVals++;
    }
  }
  printf("Possible sampling rates:");
  for (i = 0; i < numPossibleRates; i++) {
    printf(" %u", possibleRates[i]);
  }
  putchar('\n');

  snd_pcm_close(capture_handle);

  return;

}


void* handle_audio(void* nothing)
{

  int err;
  char *buffer;
  int buffer_size;
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
    int ee = snd_mixer_selem_set_capture_volume_all(elem, (*volume) * (max - min) / 100);
    if (ee != 0)
      printf("Error when setting the volume: %d\n", ee);
    long vv;
    snd_mixer_selem_get_capture_volume(elem, 1, &vv);
    printf("Ch%d --- New volume %ld (range %ld to %ld)\n", iii, vv, min, max);
    // snd_mixer_close(handle);
  }

  while (!new_config_received_flag) {

    if ((err = snd_pcm_readi (capture_handle, buffer, *buffer_frames)) != *buffer_frames) {
      fprintf (stderr, "read from audio interface failed %d (%s)\n",
               err, snd_strerror (err));
      exit (1);
    }


    pthread_mutex_lock(&audio_client_lock);
    //char* t = buffer;
    struct client* c = clients;
    struct client* previous = NULL;
    while (c != NULL) {

      // Send out the audio packet header
      easy_dsp_hdr_t header_audio = EASY_DSP_HDR_AUDIO;
      int re = write((*c).addr, &header_audio, sizeof(easy_dsp_hdr_t));

      // Now send the actual samples
      re = write((*c).addr, buffer, buffer_size);

      // Check for errors
      if (re == -1) {
        // This client is gone
        // We remove it
        if (previous == NULL) { // First client
          clients = (*c).next;
        } else {
          (*previous).next = (*c).next;
        }
      }

      // Give a warning if we wrote less bytes than we thought...
      if (re != buffer_size)
        fprintf(stdout, "Warning: less than buffer_size was written to socket.\n");

      // move on to next client
      previous = c;
      c = (*c).next;
    }
    pthread_mutex_unlock(&audio_client_lock);

  }

  free(buffer);

  fprintf(stdout, "buffer freed\n");

  snd_pcm_close (capture_handle);
  fprintf(stdout, "audio interface closed\n");

  // we are done here
  audio_thread_active_flag = 0;

  return NULL;
}

void *handle_connections_control(void* nothing) 
{
  const char *SOCKNAMEC = EASY_DSP_CONTROL_SOCKET;
  unlink(SOCKNAMEC);
  int sfd, s2;
  ssize_t len;
  struct sockaddr_un addr, remote;
  config_t audio_cfg;

  sfd = socket(AF_UNIX, SOCK_STREAM, 0);            /* Create socket */
  if (sfd == -1) {
    fprintf (stderr, "cannot create the socket control\n");
    return NULL;
  }

  memset(&addr, 0, sizeof(struct sockaddr_un));     /* Clear structure */
  addr.sun_family = AF_UNIX;                            /* UNIX domain address */
  strncpy(addr.sun_path, SOCKNAMEC, sizeof(addr.sun_path) - 1);

  if (bind(sfd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un)) == -1) {
    fprintf (stderr, "cannot bind the socket control\n");
    return NULL;
  }

  listen(sfd, 3);
  fprintf (stdout, "Bind successful control\n");

  while (1) {
    fprintf(stdout, "Waiting for a connection...\n");
    socklen_t t = sizeof(remote);
    if ((s2 = accept(sfd, (struct sockaddr *)&remote, &t)) == -1) {
      fprintf (stderr, "cannot accept the connection control\n");
      continue;
    }

    fprintf(stdout, "New client control\n");

    // Get the config
    len = recv(s2, &audio_cfg, sizeof(config_t), MSG_WAITALL);
    if (len != sizeof(config_t))
    {
      // If there is an error, just ignore it
      fprintf(stderr, "Error when receiving new config.\n");
      continue;
    }

    // Let the audio thread know that new config has been received
    new_config_received_flag = 1;

    // Wait for the audio thread to stop
    while (audio_thread_active_flag)
      ;

    // Save the new config
    *buffer_frames = audio_cfg.config.buffer_frames;
    *rate = audio_cfg.config.rate;
    *channels = audio_cfg.config.channels;
    *volume = audio_cfg.config.volume;

    // Send the new config to clients
    struct client* client;
    for (client = clients; client != NULL; client = (*client).next) {

      // First write the config magic byte
      easy_dsp_hdr_t magic_byte = EASY_DSP_HDR_CONFIG;
      write((*client).addr, &magic_byte, sizeof(magic_byte));

      // then send the config
      write((*client).addr, &audio_cfg, sizeof(config_t));

    }

    fprintf(stdout, "New configuration: %d %d %d %d", *buffer_frames, *rate, *channels, *volume);

    // Wait for a new configuration
    new_config_received_flag = 0;

    if( pthread_create(audio_thread, NULL, handle_audio, NULL) < 0) {
        perror("could not create thread to handle audio ALSA");
        return NULL;
    }
    else
      audio_thread_active_flag = 1;

  }

  return NULL;
}

void* handle_connections_audio(void* nothing) {
  const char *SOCKNAME = EASY_DSP_AUDIO_SOCKET;
  unlink(SOCKNAME);
  int sfd, s2;
  struct sockaddr_un addr, remote;
  config_t audio_cfg;

  sfd = socket(AF_UNIX, SOCK_STREAM, 0);            /* Create socket */
  if (sfd == -1) {
    fprintf (stderr, "cannot create the socket audio\n");
    return NULL;
  }

  memset(&addr, 0, sizeof(struct sockaddr_un));     /* Clear structure */
  addr.sun_family = AF_UNIX;                            /* UNIX domain address */
  strncpy(addr.sun_path, SOCKNAME, sizeof(addr.sun_path) - 1);

  if (bind(sfd, (struct sockaddr *) &addr, sizeof(struct sockaddr_un)) == -1) {
    fprintf (stderr, "cannot bind the socket\n");
    return NULL;
  }

  listen(sfd, 3);
  fprintf (stdout, "Bind successful audio\n");
  while (1) {
    fprintf(stdout, "Waiting for a connection...\n");
    socklen_t t = sizeof(remote);
    if ((s2 = accept(sfd, (struct sockaddr *)&remote, &t)) == -1) {
      fprintf (stderr, "cannot accept the connection audio\n");
      continue;
    }

    fprintf(stdout, "New client audio\n");

    // Send initial stream configuration
    audio_cfg.config.buffer_frames = *buffer_frames;
    audio_cfg.config.rate = *rate;
    audio_cfg.config.channels = *channels;
    audio_cfg.config.volume = *volume;

    easy_dsp_hdr_t magic_byte = EASY_DSP_HDR_CONFIG;
    write(s2, &magic_byte, sizeof(easy_dsp_hdr_t));
    write(s2, &audio_cfg, sizeof(config_t));

    // Send possible configurations
    magic_byte = EASY_DSP_HDR_CHANNELS;
    write(s2, &magic_byte, sizeof(easy_dsp_hdr_t));
    int configSize = sizeof(unsigned int)*numChannelConfigs;
    write(s2, &configSize, sizeof(configSize));
    write(s2, channelConfigs, configSize);

    magic_byte = EASY_DSP_HDR_RATES;
    write(s2, &magic_byte, sizeof(easy_dsp_hdr_t));
    configSize = sizeof(unsigned int)*numPossibleRates;
    write(s2, &configSize, sizeof(configSize));
    write(s2, possibleRates, configSize);


    // Create new client and add to the linked list
    struct client* new_client = malloc(sizeof(struct client));
    pthread_mutex_lock(&audio_client_lock);
    (*new_client).addr = s2;
    (*new_client).next = clients;
    clients = new_client;
    pthread_mutex_unlock(&audio_client_lock);
  }
}
