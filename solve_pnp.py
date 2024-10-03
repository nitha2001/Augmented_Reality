from est_homography import est_homography
import numpy as np


def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation
    H = est_homography(Pw[: , :-1], Pc)
    print('\n'"Pw is",Pw)
    print('\n'"H is",H)
    H1 = np.linalg.inv(K) @ H
    U, S, V = np.linalg.svd(H1[:,:-1],full_matrices=False)
    print("U is",U)
    print("V is", V)
    r = np.matmul(U, V)
    print('\n'"r is",r)
    print('\n'"S is",S)
    print("Pc is",Pc)
    L = (S[0]+S[1])/2
    t = H1[:, -1]/L
    cross_product = np.cross(r[:, 0], r[:, 1])
    cross_product=np.reshape(cross_product,[3,1])
    R = np.hstack((r, cross_product))
    print("R is", R)
    print("t is", t)
    ##### STUDENT CODE END #####
    print("Final matrices =", R.T , -R.T@t)
    return R.T, -R.T@t

def main():
    a= np.array([[1, 2],[3, 4],[1, 5],[4, 4]])
    b= np.array([[1, 2, 1],[2, 5, 7],[1, 4, 3],[1, 1, 0]])
    PnP(a, b, np.eye(3))
  

if __name__ == "__main__":
    main()
