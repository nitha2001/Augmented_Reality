import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    Pc_1 = np.concatenate([pixels ,np.ones((pixels.shape[0],1))], axis=1)
    LHS = R_wc@np.linalg.inv(K)@Pc_1.T
    L_1 = -t_wc[-1]/LHS[-1,:]
    Pw = (L_1.reshape((1,-1))*LHS) + t_wc.reshape((3,1))

    ##### STUDENT CODE END #####
    print(Pw.T)
    return Pw.T
