import serial, time
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np

class NeoPixels(object):

    def __init__(self, usb_port, colormap=None, baud_rate=115200, 
        x_range=None, num_pixels=60, update=3, normalize=True, vrange=None, range_tracking=0.):

        if colormap is None:
            colormap = cm.seismic
        if x_range is None:
            x_range = (0,2*np.pi)

        # connection to Arduino
        self.arduino = serial.Serial(usb_port, 
            baud_rate, timeout=.1)
        time.sleep(2) #give the connection some time to settle
        self.frame_count = 0
        self.update = update   # update every <this> number of frames

        # neopixel parameters
        self.num_pixels = num_pixels

        # color mapping - http://matplotlib.org/users/colormaps.html
        self.default_values = np.zeros(self.num_pixels)
        self.x_range = x_range
        self.positions = np.linspace(self.x_range[0], self.x_range[1], 
            self.num_pixels, endpoint=False)
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        self.mapper = cm.ScalarMappable(norm=norm, cmap=colormap)

        # normalization setup
        self.normalize = normalize
        self.vrange = vrange if vrange is not None else np.array([0., 1.])
        self.range_tracking = range_tracking
        if self.range_tracking > 0:
            self.vrange[1] = 0.
        self.norm_start_count = 0


        self.lightify()

    def lightify(self, vals=None, realtime=False):

        if realtime is True:    # cannot update every frame
            self.frame_count += 1
            if (self.frame_count%self.update) != 0:
                return

        if vals is None:
            pixel_values = self.default_values
        else:
            # First reverse the array since the LED are clockwise,
            # and the direction of arrivals counterclockwise...
            vals = vals[::-1]

            # interpolate and normalize

            if len(vals) != self.num_pixels:
                xvals = np.linspace(self.x_range[0], self.x_range[1], 
                    len(vals), endpoint=False)
                pixel_values = np.interp(self.positions, xvals, vals)
            else:
                pixel_values = vals

            if self.range_tracking > 0.:
                new_range = np.array([np.min(vals), np.max(vals)])
                if self.norm_start_count < 100:
                    self.norm_start_count += 1
                    self.vrange = ((self.norm_start_count - 1) * self.vrange + new_range) / self.norm_start_count
                else:
                    self.vrange = (
                            self.range_tracking  * self.vrange
                            + (1 - self.range_tracking) * new_range
                            )

            if self.normalize:
                pixel_values -= self.vrange[0]
                pixel_values /= self.vrange[1] - self.vrange[0]

            pixel_values[pixel_values < 0.] = 0.
            pixel_values[pixel_values > 1.] = 1.

        '''
        colors = self.mapper.to_rgba(pixel_values)
        colors = colors[:,:3]
        '''

        # nice map ?
        # It is good to have slightly less intensity
        # for lower sound intensity
        colors = np.zeros((len(pixel_values), 3))
        colors[:,0] = pixel_values
        colors[:,1] = 0.05 * (1. - pixel_values)
        colors[:,2] = 0.1 * (1. - pixel_values)
        
        # Send to the arduino
        colors = np.reshape((colors * 254).round().astype(np.uint8),-1)
        self.arduino.write(colors.tobytes())

    def lightify_mono(self, rgb=[255,0,0], realtime=False):

        if realtime is True:    # cannot update every frame
            self.frame_count += 1
            if (self.frame_count%self.update) != 0:
                return

        pixel_values = np.array(rgb)/255
        pixel_values = np.tile(pixel_values,(60,1))
        colors = np.reshape((pixel_values*255).round().astype(np.uint8),-1)
        self.arduino.write(colors.tobytes())






