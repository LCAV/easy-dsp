import numpy as np
import scipy.linalg as la

# As 'measured' by Juan
'''
R_pyramic = np.array([
    [-1.756492069,  -3.042333507,   1.139761726],  # 0
    [-2.91789798,   -5.053947553,   4.396223799],
    [-4.079303892,  -7.0655616, 7.652685873],
    [-4.543866256,  -7.870207219,   8.955270702],
    [-4.776147439,  -8.272530028,   9.606563117],
    [-5.240709803,  -9.077175647,   10.90914795],
    [-6.402115715,  -11.08878969,   14.16561002],
    [-7.563521626,  -13.10040374,   17.42207209],  # 7

    [-6.4,  -10,    21.3],  # 8
    [-6.4,  -6,     21.3],
    [-6.4,  -2,     21.3],
    [-6.4,  -0.4,   21.3],
    [-6.4,  0.4,    21.3],
    [-6.4,  2,      21.3],
    [-6.4,  6,      21.3],
    [-6.4,  10,     21.3],  # 15

    [3.512984138,   0,  1.139761726],  # 16
    [5.835795961,   0,  4.396223799],
    [8.158607784,   0,  7.652685873],
    [9.087732513,   0,  8.955270702],
    [9.552294877,   0,  9.606563117],
    [10.48141961,   0,  10.90914795],
    [12.80423143,   0,  14.16561002],
    [15.12704325,   0,  17.42207209],  # 23

    [13.70043101,   -2.5,    21.3],  # 24
    [10.2363294,    -4.5,    21.3],
    [6.772227782,   -6.5,    21.3],
    [5.386587136,   -7.3,    21.3],
    [4.693766812,   -7.7,    21.3],
    [3.308126166,   -8.5,    21.3],
    [-0.155975449,  -10.5,   21.3],
    [-3.620077064,  -12.5,   21.3],  # 31

    [-1.756492069,  3.042333507,    1.139761726],  # 32
    [-2.91789798,   5.053947553,    4.396223799],
    [-4.079303892,  7.0655616,  7.652685873],
    [-4.543866256,  7.870207219,    8.955270702],
    [-4.776147439,  8.272530028,    9.606563117],
    [-5.240709803,  9.077175647,    10.90914795],
    [-6.402115715,  11.08878969,    14.16561002],
    [-7.563521626,  13.10040374,    17.42207209],  # 39

    [-3.620077064,  12.5,   21.3],  # 40
    [-0.155975449,  10.5,   21.3],
    [3.308126166,   8.5,    21.3],
    [4.693766812,   7.7,    21.3],
    [5.386587136,   7.3,    21.3],
    [6.772227782,   6.5,    21.3],
    [10.2363294,    4.5,    21.3],
    [13.70043101,   2.5,    21.3],  # 47

    ]).T / 100.
'''

x = 0.27 + 2*0.015  # length of one side
c1 = 1./np.sqrt(3.)
c2 = np.sqrt(2./3.)
c3 = np.sqrt(3.)/6.
c4 = 0.5
corners = np.array( [
    [ 0, x*c1, -x*c3, -x*c3,],
    [ 0,   0.,  x*c4, -x*c4,],
    [ 0, x*c2,  x*c2,  x*c2,],
    ])

# relative placement of microphones on one pcb
pcb = np.array([-0.100, -0.060, -0.020, -0.004, 0.004, 0.020, 0.060, 0.100])

def line(p1, p2, dist):
    # Places points at given distance on the line joining p1 -> p2, starting at the midpoint
    o = (p1 + p2) / 2.
    u = (p2 - p1) / la.norm(p2 - p1)

    pts = []
    for d in dist:
        pts.append(o + d*u)

    return pts

R_pyramic = np.array(
        line(corners[:,0], corners[:,3], pcb) +
        line(corners[:,3], corners[:,2], pcb) +
        line(corners[:,0], corners[:,1], pcb) +
        line(corners[:,1], corners[:,3], pcb) +
        line(corners[:,0], corners[:,2], pcb) +
        line(corners[:,2], corners[:,1], pcb)
        ).T

# Reference point is 1cm below zero'th mic
R_pyramic[2,:] += 0.01 - R_pyramic[2,0]

def get_pyramic(dim=3):
    '''
    Get a copy of the pyramic microphone location

    A single optional parameter `dim` can be given.
    `dim=3` returns the full array
    `dim=2` returns the top part of the array with the 3rd dimension ommitted
    '''
    if dim == 3:
        return R_pyramic.copy(), list(range(R_pyramic.shape[1]))
    elif dim == 2:
        I = list(range(8, 16)) + list(range(24, 32)) + list(range(40, 48))
        return R_pyramic[:,I].copy(), I

if __name__ == "__main__":

    from experiment import PointCloud

    pyramic = PointCloud(X=R_pyramic)

    D = np.sqrt(pyramic.EDM())
    print(D[0,16])

    pyramic.plot()

