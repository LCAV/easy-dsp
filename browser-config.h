#ifndef _BROWSER_CONFIG_H_
#define _BROWSER_CONFIG_H_

// Local sockets
#define EASY_DSP_CONTROL_SOCKET      "/tmp/micros-control.socket"
#define EASY_DSP_AUDIO_SOCKET        "/tmp/micros-audio.socket"
#define EASY_DSP_STREAM_CONTROL_SOCKET  "/tmp/micros-stream-control.socket"

// Server configurations
#define EASY_DSP_WSAUDIO_IP_ADDR      "0.0.0.0" // 0.0.0.0 = all IPv4 addresses on the local machine
#define EASY_DSP_WSAUDIO_SERVER_PORT  "7321"
#define EASY_DSP_WSCONFIG_IP_ADDR     "0.0.0.0" // 0.0.0.0 = all IPv4 addresses on the local machine
#define EASY_DSP_WSCONFIG_SERVER_PORT "7322"

// Audio configuration
#define EASY_DSP_VOLUME                     (80)
#define EASY_DSP_NUM_CHANNELS               (6)
#define EASY_DSP_AUDIO_FREQ_HZ              (48000)
#define EASY_DSP_AUDIO_FORMAT_BITS          (16) // NOT changeable!
#define EASY_DSP_AUDIO_FORMAT_BYTES         (EASY_DSP_AUDIO_FORMAT_BITS / 8)
#define EASY_DSP_BUFFER_SIZE_BYTES          (4096)

// Stream magic header
#define EASY_DSP_HDR_AUDIO                  (0x0)
#define EASY_DSP_HDR_CONFIG                 (0x1)

#endif /* _BROWSER_CONFIG_H_ */
