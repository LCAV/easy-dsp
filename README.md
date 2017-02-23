# EasyDSP --- Interface for Audio Control of Embedded Microphone Arrays

EasyDSP is modular and multiplatform interface for microphone arrays based on
embedded Linux platforms. It provides easy access to the audio samples
collected by the microphone array and let the user run real-time processing on
the stream.  The result of processing can be visualized live in the browser.

EasyDSP is based on web technologies, namely websockets and javascript, for the
streaming of the samples and the user interface. Python is used for the
processing of the audio samples.

Currently ALSA drivers for the microphone array on the embedded platform are required.
EasyDSP has been developed for the [CompactSix](http://github.com/LCAV/CompactSix) array
based on the Beaglebone Black single board computer.
Due to the modular structure of the architecture, we do not forsee problems when porting
to other platforms. A [fork](http://github.com/sahandKashani/easy-dsp) is being modified
to work with the [Pyramic](https://github.com/LCAV/Pyramic) platform with 48 microphones
and based on FPGA technology.

## Documentation

The documentation is in `docs/` (the Markdown source is in `docs-source/`) or at this address: https://lcav.github.io/easy-dsp/.

## Quick Install

Connect to the microphone array via ssh. Install [libwebsock](https://github.com/payden/libwebsock)
and [Jansson](http://www.digip.org/jansson/) and run the following commands as root.

    apt-get install apache2 libapache2-mod-php5 php5 php5-common
    cd /var
    git clone https://github.com/LCAV/easy-dsp
    cd easy-dsp
    touch logs.txt
    chown www-data:www-data logs.txt
    cp microphones.virtualhost /etc/apache2/sites-available/microphones
    a2ensite microphones
    echo "Listen 8081" >> /etc/apache2/ports.conf
    usermod -aG audio www-data
    setfacl -m u:www-data:rw /dev/snd/*
    rm /tmp/micros-audio.socket /tmp/micros-control.socket
    service apache2 restart
    make

See the [docs](https://lcav.github.io/easy-dsp/getting-started/) for the rest of the instructions.

## Authors

* Basile Bruneau
* Eric Bezzam
* Robin Scheibler

## License

    Copyright 2016, 2017 Basile Bruneau, Eric Bezzam, Robin Scheibler

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
