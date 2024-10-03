from math import sqrt
import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####
    a = np.linalg.norm(Pw[1, :] - Pw[2, :])
    b = np.linalg.norm(Pw[0, :] - Pw[2, :])
    c = np.linalg.norm(Pw[0, :] - Pw[1, :])
    print('a,b,c',a,'\n',b,'\n',c)
    
    #Unit Vectors
    j_1 = np.array([(Pc[0,0]-K[0,2])/K[0,0],(Pc[0,1]-K[1,2])/K[0,0],1])
    j_2 = np.array([(Pc[1,0]-K[0,2])/K[0,0],(Pc[1,1]-K[1,2])/K[0,0],1])
    j_3 = np.array([(Pc[2,0]-K[0,2])/K[0,0],(Pc[2,1]-K[1,2])/K[0,0],1])

    j1 = j_1/np.linalg.norm(j_1)
    j2 = j_2/np.linalg.norm(j_2)
    j3 = j_3/np.linalg.norm(j_3)

    print('j1,j2,j3',j1,'\n',j2,'\n',j3)

    # Alpha, beta and gamma
    C_A = np.dot(j2,j3)
    C_B = np.dot(j1,j3)
    C_G = np.dot(j1,j2)
    print('C_A,C_B,C_G',C_A,'\n',C_B,'\n',C_G)

    # squared terms 
    asq = a**2
    bsq = b**2
    csq = c**2
    print('asq,bsq,csq',asq,'\n',bsq,'\n',csq)

    # Coeffecients
    A0 = ((1+((a**2-c**2)/b**2))**2) - ((4*(a**2)/b**2) * C_G ** 2)
    A1 = 4*((-((asq-csq)/bsq)*(1+((asq-csq)/bsq))*C_B)+(2*asq*C_G**2*C_B/bsq)-((1-((asq+csq)/bsq))*C_A*C_G))
    A2 = 2*((((asq-csq)/bsq)**2) - 1 + (2*((asq-csq)/bsq)**2*C_B**2) + (2*(((bsq-csq)/bsq)*C_A**2)) - (4*((asq+csq)/bsq)*C_A*C_B*C_G) + (2*((bsq-asq)/bsq*C_G**2)))
    A3 = 4*(((asq-csq)/bsq*(1-((asq-csq)/bsq))*C_B)-((1-((asq+csq)/bsq))*C_A*C_G)+(2*(csq/bsq)*C_A**2*C_B))
    A4 = (((asq-csq)/bsq)-1)**2 - (4*(csq/bsq)*C_A**2)
    print('A0,A1,A2,A3,A4',A0,'\n',A1,'\n',A2,'\n',A3,'\n',A4)

    # Finding roots
    coeff = [A4,A3,A2,A1,A0]
    v = np.roots(coeff)
    v_real = []
    for i in v:
       if np.isreal(i):
        v_real.append(np.real(i))

    v_real = np.reshape(v_real,(2,1))
    print("v_real is", v_real)

    i = 0
    p1=[]
    p2=[]
    p3=[]

    s1=[]
    s2=[]
    s3=[]
    u = np.zeros_like(v_real)
    
    for i in range(u.shape[0]):
        u[i] = (((-1+((asq-csq)/bsq))*v_real[i]**2)-(2*((asq-csq)/bsq)*C_B*v_real[i])+1+((asq-csq)/bsq))/(2*(C_G-(v_real[i]*C_A)))
 
        s1.append(sqrt(csq/(1+(u[i]**2)-(2*u[i]*C_G))))
        s2.append(u[i]*s1[i])
        s3.append(v_real[i]*s1[i])
        p1.append(s1[i]*j1)
        p2.append(s2[i]*j2)
        p3.append(s3[i]*j3)
    
    print('u is', u)
    print('s is',s1,s2,s3)
    print("p1 is",p1)
    print('p2 is', p2)
    print('p3 is',p3)

    
    min = 100000000000000000000
    for i in range(len(p1)):
        Pc_3d = np.array([p1[i], p2[i], p3[i]])
        R, t = Procrustes(Pc_3d, Pw[0:3])
        q = np.matmul(K, np.matmul(R.T, Pw[3])-R.T@t)
        q1 = q/q[2]
        Pc_1 = np.concatenate([Pc,np.ones((Pc.shape[0],1))],axis=1)
        if np.linalg.norm(Pc_1[3]-q1)<min:
            min = np.linalg.norm(Pc_1[3]-q1)
            R_t = R
            t_t = t

        

    print('R is',R,'\n','t is',t)
    print('K is',K)
    return R_t,t_t

             
          

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    centroid_X = np.mean(X, axis=0)
    centroid_Y = np.mean(Y, axis=0)

    X_centered = X - centroid_X
    Y_centered = Y - centroid_Y

    # SVD
    U, _, VT = np.linalg.svd(np.dot(Y_centered.T, X_centered))
    R_optimal = np.dot(U, VT)

    # Check for reflection (determinant of R)
    if np.linalg.det(R_optimal) < 0:
        # If determinant is -1, it's a reflection. Correct it by flipping one axis.
        VT[-1] *= -1
        R_optimal = np.dot(U, VT)

    t_optimal = centroid_Y - np.dot(R_optimal, centroid_X)

    return R_optimal, t_optimal

def main():
    a= np.array([[1, 2],[3, 4],[1, 5],[4, 4]])
    b= np.array([[1, 2, 1],[2, 5, 7],[1, 4, 3],[1, 1, 0]])
    P3P(a, b, np.eye(3))
  

if __name__ == "__main__":
    main()

