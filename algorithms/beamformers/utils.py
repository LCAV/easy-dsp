import numpy as np

def polar2cart(polar):
    """
    Convert polar (radians) to cartesian coodinates.

    Parameters
    -----------
    polar : numpy array (float)
        Polar coordinates (distance, angle) where each column is a point. Angle in radians.

    Returns
    -----------
    cart : numpy array
        Cartesian coordinates (x,y) where each column is a point.

    """

    # obtain coordinates
    if len(polar.shape)==1:
        r = polar[0]
        theta = polar[1]
    else:
        r = polar[0,:]      # radius
        theta = polar[1,:]  # angle

    # convert to cartesian
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if len(polar.shape)==1:
        cart = np.array([x, y])
    else:
        cart = np.zeros([polar.shape[0], polar.shape[1]])
        cart[0,:] = x
        cart[1,:] = y
    cart[abs(cart)<tol] = 0
    return cart

def cart2polar(cart):
    """
    Convert cartesian to polar (radians) coodinates.

    Parameters
    -----------
    cart : numpy array (float)
        Cartesian coordinates (x,y) where each column is a point.

    Returns
    -----------
    polar : numpy array
        Polar coordinates (distance, angle) where each column is a point.

    """

    if len(cart.shape)==1:
        # obtain coordinates
        x = cart[0]
        y = cart[1]
        # convert to polar
        r = np.sqrt(x**2 + y**2)    # distance
        theta = np.arctan2(y,x)
        # store in output
        if abs(r) < tol:
            r = 0
        if abs(theta) < tol:
            theta = 0
        polar = np.array([r, theta])
    else:
        # obtain coordinates
        x = cart[0,:]
        y = cart[1,:]
        # convert to polar
        r = np.sqrt(x**2 + y**2)    # distance
        theta = np.arctan2(y,x)
        # store in output
        polar = np.zeros([cart.shape[0], cart.shape[1]])
        polar[0,:] = r
        polar[1,:] = theta
        for i in range(len(polar[0,:])):
            if abs(polar[0,i]) < tol:
                polar[0,i] = 0
            if abs(polar[1,i]) < tol:
                polar[1,i] = 0

    return polar
