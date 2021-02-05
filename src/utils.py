import numpy as np

'''
General Utility functions
'''
def divide_limits(a,b,n):
    """
    Divide limit [a,b] in n parts
    """
    l=[]
    l.append(a)
    val = (b-a)/n
    
    temp=a
    while temp <= b:
        temp+=val
        l.append(temp)
        
    l.append(b)
    return l

def flip_quat(quat):
    '''
    Flip the w term to the front
    Args (nparray): a quaternion with w as last term
    Returns (nparray): quaternion [w, x, y, z]
    '''
    w = quat[3]
    x = quat[0]
    y = quat[1]
    z = quat[2]
    return np.array([w,x,y,z])

def reverse_flip_quat(quat):
    '''
    Flip the w term to the end
    Args (nparray): a quaternion with w as first term
    Returns (nparray): quaternion [x,y,z,w]
    '''
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    return np.array([x,y,z,w])

def rotate_vect(v, angle):
    '''
    Rotate a vector
    Args (nparray, angle): a 2D vector & angle 
    Returns (nparray): rotated vector 
    '''
    c_a = np.cos(angle)
    s_a = np.sin(angle)
    return np.array([c_a*v[0] - s_a*v[1], s_a*v[0] + c_a*v[1], v[2]])

