// http://arduino.stackexchange.com/questions/1013/how-do-i-split-an-incoming-string
// http://internetofhomethings.com/homethings/?p=927
// http://www.hobbytronics.co.uk/arduino-serial-buffer-size--> increase buffer size?

#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
#include <avr/power.h>
#endif

#include <limits.h>

// Turn off the LEDs after a timeout
#define TIMEOUT 2000

unsigned long last_update = 0;
char is_on = 0;

#define PIN 6
#define INPUT_SIZE 180
#define NUM_LEDS 60

char input[INPUT_SIZE + 1];
Adafruit_NeoPixel strip = Adafruit_NeoPixel(NUM_LEDS, PIN, NEO_GRB + NEO_KHZ800);

void reset_leds()
{
  // Turn off all the LEDs
  uint8_t i;
  for (i = 0 ; i < NUM_LEDS ; i++)
    strip.setPixelColor(i, strip.Color(0, 0, 0));
  strip.show();

  is_on = 0;
}

unsigned long time_since_last_update()
{
  unsigned long now = millis();

  if (now < last_update)
    return ULONG_MAX - last_update + now;
  else
    return now - last_update;
}

void setup() {
  Serial.begin(115200);

  strip.begin();
  
  reset_leds(); // Initialize all pixels to 'off'
}

void loop() {

  if (Serial.available() > 0) {

    last_update = millis();
    is_on = 1;

    byte size = Serial.readBytes(input, INPUT_SIZE);
    input[size] = 0;

    uint8_t pixelIdx, red, green, blue;
    for ( pixelIdx = 0; pixelIdx < NUM_LEDS; pixelIdx++)
    {
      red = uint8_t(input[pixelIdx * 3]);
      green = uint8_t(input[pixelIdx * 3 + 1]);
      blue = uint8_t(input[pixelIdx * 3 + 2]);
      strip.setPixelColor(pixelIdx, strip.Color(red, green, blue));
    }
    strip.show();
  }

  // Turn off LEDs on timeout
  if (is_on && time_since_last_update() > TIMEOUT)
    reset_leds();
}
