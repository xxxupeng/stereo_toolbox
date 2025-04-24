import math
import numpy as np

def eulerAnglesToRotationMatrix(theta):
    cos_theta_0 = math.cos(theta[0])
    sin_theta_0 = math.sin(theta[0])
    cos_theta_1 = math.cos(theta[1])
    sin_theta_1 = math.sin(theta[1])
    cos_theta_2 = math.cos(theta[2])
    sin_theta_2 = math.sin(theta[2])
    R_x = np.array([[1, 0, 0],
                    [0, cos_theta_0, -sin_theta_0],
                    [0, sin_theta_0, cos_theta_0]
                    ])

    R_y = np.array([[cos_theta_1, 0, sin_theta_1],
                    [0, 1, 0],
                    [-sin_theta_1, 0, cos_theta_1]
                    ])

    R_z = np.array([[cos_theta_2, -sin_theta_2, 0],
                    [sin_theta_2, cos_theta_2, 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def PinholeEulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R