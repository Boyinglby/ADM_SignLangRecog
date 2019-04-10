# -*- coding: utf-8 -*-
"""
Geometric transformations on 3D point cloud.
Created on Wed Apr 10 11:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/sign-language

"""


import copy
import math
import numpy as np


class Transforms3D(object):
    
    @staticmethod
    def rotate(pointcloud):
        assert pointcloud.__class__.__name__ == 'PointCloud3D' \
        and not pointcloud._data is None
        
        pc = copy.deepcopy(pointcloud)
        for frame, joints in enumerate(pc._data):
            # Reshape joints into Nx3 matrix.
            joints = joints.reshape(-1, 3)
            
            # Get left shoulder L, right shoulder R and spine center C.
            L = joints[pointcloud.JointType_ShoulderLeft, :]
            R = joints[pointcloud.JointType_ShoulderRight, :]
            C = joints[pointcloud.JointType_SpineMid, :]
            
            # Calculate unit vector n along the normal to LRC-plane.
            CL = L - C
            CR = R - C
            n = np.cross(CL, CR) / (np.linalg.norm(CL) * np.linalg.norm(CR))
            
            # Calculate angle between the projection
            # of n on XZ-plane and Z-axis.
            n = n * np.array([1, 0, 1])
            k = np.array([0, 0, -1])
            ratio = np.dot(n, k) / (np.linalg.norm(n) * np.linalg.norm(k))
            theta = np.arccos(ratio)
            theta = math.copysign(theta, n[0])
            
            # Construct transformation matrix R for rotation around Y-axis.
            R = np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
            
            # Perform rotation and update data.
            joints = np.dot(R, joints.T).T
            pc._data[frame] = joints.reshape(-1)
        
        return pc
    
    @staticmethod
    def translate(pointcloud):
        assert pointcloud.__class__.__name__ == 'PointCloud3D' \
        and not pointcloud._data is None
        
        pc = copy.deepcopy(pointcloud)
        for frame, joints in enumerate(pc._data):
            # Reshape joints into Nx3 matrix.
            joints = joints.reshape(-1, 3)
            
            # Construct transformation matrix T for translation.
            C = joints[pointcloud.JointType_SpineMid, :]
            T = -C
            
            # Perform translation and update data.
            joints += T
            pc._data[frame] = joints.reshape(-1)
        
        return pc
    
    @staticmethod
    def transform(pointcloud):
        assert pointcloud.__class__.__name__ == 'PointCloud3D' \
        and not pointcloud._data is None
        
        return Transforms3D.translate(Transforms3D.rotate(pointcloud))
