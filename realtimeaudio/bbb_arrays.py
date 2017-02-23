import numpy as np

# microphone array positions
n_mics = 6        # number of microphones
radius = 0.0625     # array radius in mm

# angle locations of microphones in fractions of 2 \pi
angles = np.array([0., 2., 4., 1., 3., 5.]) * 1./6.

# Array of microphones location sin 3D
R_compactsix_circular_1 = np.array([ 
  radius*np.cos(2.*np.pi*angles), 
  radius*np.sin(2.*np.pi*angles), 
  np.zeros(n_mics),
  ])

R_compactsix_random = np.array(
    [[51.816, 64.516, 24.13,  84.582, 44.45,  25.146],
     [36.068, 10.668, 16.002, 16.764, 10.414, 33.528],
     [ 0.0,    0.0,    0.0,    0.0,    0.0,    0.0  ]]) * 1e-3
