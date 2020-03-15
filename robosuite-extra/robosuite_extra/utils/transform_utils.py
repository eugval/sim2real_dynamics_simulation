from robosuite.utils.transform_utils import *
import numpy as np



def eulerx2mat(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[1.,0.,0.],
                     [0.,cos ,-sin],
                     [0.,sin, cos]])

def eulery2mat(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos, 0., sin],
                     [0.,1.,0.],
                     [-sin, 0., cos]])

def eulerz2mat(angle):
    cos = np.cos(angle)
    sin = np.sin(angle)
    return np.array([[cos,-sin, 0.],
                     [sin, cos, 0.],
                     [0.,0.,1.]])


def euler2mat(euler_x,euler_y, euler_z):
    r_x = eulerx2mat(euler_x)
    r_y = eulery2mat(euler_y)
    r_z = eulerx2mat(euler_z)
    return  r_z.dot(r_y.dot(r_x))

