#ifndef _BROWSER_CONFIG_H_
#define _BROWSER_CONFIG_H_

// Server configurations
#define EASY_DSP_WSAUDIO_IP_ADDR      "0.0.0.0" // 0.0.0.0 = all IPv4 addresses on the local machine
#define EASY_DSP_WSAUDIO_SERVER_PORT  "7321"
#define EASY_DSP_WSCONFIG_IP_ADDR     "0.0.0.0" // 0.0.0.0 = all IPv4 addresses on the local machine
#define EASY_DSP_WSCONFIG_SERVER_PORT "7322"

// Audio configuration
#define EASY_DSP_VOLUME                     (80)
#define EASY_DSP_NUM_CHANNELS               (6)
#define EASY_DSP_AUDIO_FREQ_HZ              (48000)
#define EASY_DSP_AUDIO_FORMAT_BITS          (16) // cannot be changed!
#define EASY_DSP_AUDIO_FORMAT_BYTES         (EASY_DSP_AUDIO_FORMAT_BITS / 8)
#define EASY_DSP_BUFFER_SIZE_BYTES          (4096)

// configuration parameters
extern unsigned int buffer_frames;
extern unsigned int rate;
extern unsigned int volume;
extern unsigned int channels;

// possible alsa configurations
extern unsigned int* channelConfigs;
extern unsigned int numChannelConfigs;
extern unsigned int* possibleRates;
extern unsigned int numPossibleRates;

// Define a union to access config in different ways
typedef union {
  uint8_t bytes[4*sizeof(uint32_t)];
  uint32_t words[4];
  struct {
    unsigned int buffer_frames;
    unsigned int rate;
    unsigned int channels;
    unsigned int volume;
  } config;
}
config_t;

void set_config(config_t*);

#endif /* _BROWSER_CONFIG_H_ */

