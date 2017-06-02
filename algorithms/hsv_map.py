import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def make_hsv_map(background, source, N=256, name='cmyk_to_rgb'):

    background_rgb = colorsys.hsv_to_rgb(*background)
    source_rgb = colorsys.hsv_to_rgb(*source)

    return LinearSegmentedColormap.from_list(name, [background_rgb, source_rgb], N=N)
