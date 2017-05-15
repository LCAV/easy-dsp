/*
  The Main Daemon
  gcc -o browser-main-daemon -lasound browser-main-daemon.c
  ./browser-main-daemon
*/

#include <stdio.h>
#include <stdlib.h>
#include <alsa/asoundlib.h>
#include <sys/un.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <unistd.h>
#include <stdint.h>

#include "global.h"
#include "browser-config.h"
#include "browser-wsconfig.h"
#include "browser-wsaudio.h"

#define ARRAY_SIZE(a) (sizeof(a) / sizeof *(a))

void sig_handler(int signo) {
  if (signo == SIGPIPE) {
    fprintf(stdout, "received SIGPIPE\n");
  }
}

struct client {
  int addr;
  struct client* next;
};

void query_available_config(void);
void* handle_connections_audio(void* nothing);
void* handle_connections_control(void* nothing);
void* handle_audio(void* nothing);

struct client* clients;
pthread_t* audio_thread;
pthread_mutex_t audio_client_lock;

// alsa parameters
snd_pcm_t *capture_handle;

// a few flags
int audio_thread_active_flag = 0;
int new_config_received_flag = 0;

// configuration parameters
unsigned int buffer_frames = EASY_DSP_BUFFER_SIZE_BYTES;
unsigned int rate = EASY_DSP_AUDIO_FREQ_HZ;
unsigned int volume = EASY_DSP_VOLUME;
unsigned int channels = EASY_DSP_NUM_CHANNELS;

// possible alsa configurations
unsigned int* channelConfigs;
unsigned int numChannelConfigs = 0;
unsigned int* possibleRates;
unsigned int numPossibleRates= 0;


int main(void) {
  audio_thread = malloc(sizeof(*audio_thread));
  clients = NULL;

  query_available_config();

  channels = channelConfigs[0];
  rate = possibleRates[0];

  // "catch" SIGPIPE we get when we try to send data to a disconnected client
  if (signal(SIGPIPE, sig_handler) == SIG_ERR) {
    printf("\ncan't catch SIGPIPE\n");
  }

  pthread_t connections_audio_thread;
  pthread_t connections_control_thread;

  if( pthread_create(&connections_audio_thread, NULL, handle_connections_audio, 
    NULL) < 0) {
      perror("could not create thread to handle connections audio");
      return 1;
  }

  if( pthread_create(&connections_control_thread, NULL, 
    handle_connections_control, NULL) < 0) {
      perror("could not create thread to handle connections control");
      return 1;
  }

  if( pthread_create(audio_thread, NULL, handle_audio, NULL) < 0) {
      perror("could not create thread to handle audio ALSA");
      return 1;
  }
  else
    audio_thread_active_flag = 1;  // Let the world know we are active

  while (true) {
      // Sleep for a long time to not take CPU cycles. ANY constant could work
      // here.
      sleep(10);
  }

  exit (0);
}

void query_available_config(void) {
  unsigned int i;
  int err;
  unsigned int minval, maxval;
  unsigned int numReadVals;
  static const unsigned int rates[] = {5512, 8000, 11025, 16000, 22050, 32000, 
    44100, 48000, 64000, 88200, 96000, 176400, 192000,
  };

  const char *device_name = "hw:0";
  snd_pcm_hw_params_t *hw_params;

  err = snd_pcm_open(&capture_handle, device_name, SND_PCM_STREAM_CAPTURE, 0);
  if (err < 0) {
      fprintf(stderr, "cannot open device '%s': %s\n", device_name, 
        snd_strerror(err));
      return;
  }

  snd_pcm_hw_params_alloca(&hw_params);
  err = snd_pcm_hw_params_any(capture_handle, hw_params);
  if (err < 0) {
      fprintf(stderr, "cannot get hardware parameters: %s\n", 
        snd_strerror(err));
      snd_pcm_close(capture_handle);
      return;
  }

  // query possible channel config
  numReadVals = 0;
  err = snd_pcm_hw_params_get_channels_min(hw_params, &minval);
  if (err < 0) {
      fprintf(stderr, "cannot get minimum channels count: %s\n", 
        snd_strerror(err));
      snd_pcm_close(capture_handle);
      return;
  }
  err = snd_pcm_hw_params_get_channels_max(hw_params, &maxval);
  if (err < 0) {
      fprintf(stderr, "cannot get maximum channels count: %s\n", 
        snd_strerror(err));
      snd_pcm_close(capture_handle);
      return;
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


void* handle_audio(void* nothing) {
  int err;
  char *buffer;
  int buffer_size;
  snd_pcm_hw_params_t *hw_params;
	snd_pcm_format_t format = SND_PCM_FORMAT_S16_LE;

  if ((err = snd_pcm_open (&capture_handle, "hw:0", 
    SND_PCM_STREAM_CAPTURE, 0)) < 0) {
    fprintf (stderr, "cannot open audio device %s (%s)\n",
             "hw:0", snd_strerror (err));
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

  if ((err = snd_pcm_hw_params_set_access (capture_handle, hw_params, 
    SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
    fprintf (stderr, "cannot set access type (%s)\n", snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params access set\n");

  if ((err = snd_pcm_hw_params_set_format (capture_handle, hw_params, 
    format)) < 0) {
    fprintf (stderr, "cannot set sample format (%s)\n", snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params format set\n");


  if ((err = snd_pcm_hw_params_set_rate_near (capture_handle, hw_params, &rate, 
    0)) < 0) {
    fprintf (stderr, "cannot set sample rate (%s)\n", snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params rate set\n");

  if ((err = snd_pcm_hw_params_set_channels (capture_handle, hw_params, 
    channels)) < 0) {
    fprintf (stderr, "cannot set channel count (%s)\n", snd_strerror (err));
    exit (1);
  }

  fprintf(stdout, "hw_params channels set\n");

  if ((err = snd_pcm_hw_params (capture_handle, hw_params)) < 0) {
    fprintf (stderr, "cannot set parameters (%s)\n", snd_strerror (err));
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

  buffer_size = buffer_frames * snd_pcm_format_width(format) / 8 * (channels);
  if ((buffer = (char*) malloc(buffer_size)) == NULL) {
    fprintf(stdout, "Cannot allocate buffer %p (size: %d)\n", buffer, 
      buffer_size);
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
    int ee = snd_mixer_selem_set_capture_volume_all(elem, (volume) * 
      (max - min) / 100);
    if (ee != 0)
      printf("Error when setting the volume: %d\n", ee);
    long vv;
    snd_mixer_selem_get_capture_volume(elem, 1, &vv);
    printf("Ch%d --- New volume %ld (range %ld to %ld)\n", iii, vv, min, max);
    // snd_mixer_close(handle);
  }

  while (!new_config_received_flag) {

    if ((err = snd_pcm_readi (capture_handle, buffer, buffer_frames)) != 
      (int)buffer_frames) {
      fprintf (stderr, "read from audio interface failed %d (%s)\n",
               err, snd_strerror (err));
      exit (1);
    }


    pthread_mutex_lock(&audio_client_lock);

    send_audio(buffer);

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


void *handle_connections_control(void* nothing) {
  wsconfig_main();
  while (1);
  return NULL;
}


void set_config (config_t* new_audio_cfg) {
  // Let the audio thread know that new config has been received
  new_config_received_flag = 1;

  // Wait for the audio thread to stop
  while (audio_thread_active_flag);
  fprintf(stdout, "Audio thread momentarily stopped for setting new \
    parameters.\n");

  // Save the new config
  buffer_frames = new_audio_cfg->config.buffer_frames;
  rate = new_audio_cfg->config.rate;
  channels = new_audio_cfg->config.channels;
  volume = new_audio_cfg->config.volume;

  fprintf(stdout, "New configuration: %d %d %d %d\n", buffer_frames, rate, 
    channels, volume);

  send_new_audio_config();

  // Wait for a new configuration
  new_config_received_flag = 0;

  if( pthread_create(audio_thread, NULL, handle_audio, NULL) < 0) {
      perror("could not create thread to handle audio ALSA");
      return;
  }
  else
    audio_thread_active_flag = 1;

  return;
}


void* handle_connections_audio(void* nothing) {
  wsaudio_main();
  while (1);
  return NULL;
}


