# -*- coding: utf-8 -*-
"""
Point Cloud.
Created on Sun Apr  7 11:00:00 2019
Author: Prasun Roy | CVPRU-ISICAL (http://www.isical.ac.in/~cvpr)
GitHub: https://github.com/prasunroy/sign-language

"""


from __future__ import division
from matplotlib import animation as anim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import ndarray
import os


class JointType(object):
    r"""Enumerates all joint types."""
    
    def __init__(self):
        self.JointType_SpineBase = 0
        self.JointType_SpineMid = 1
        self.JointType_Neck = 2
        self.JointType_Head = 3
        self.JointType_ShoulderLeft = 4
        self.JointType_ElbowLeft = 5
        self.JointType_WristLeft = 6
        self.JointType_HandLeft = 7
        self.JointType_ShoulderRight = 8
        self.JointType_ElbowRight = 9
        self.JointType_WristRight = 10
        self.JointType_HandRight = 11
        self.JointType_HipLeft = 12
        self.JointType_KneeLeft = 13
        self.JointType_AnkleLeft = 14
        self.JointType_FootLeft = 15
        self.JointType_HipRight = 16
        self.JointType_KneeRight = 17
        self.JointType_AnkleRight = 18
        self.JointType_FootRight = 19
        self.JointType_SpineShoulder = 20
        self.JointType_HandTipLeft = 21
        self.JointType_ThumbLeft = 22
        self.JointType_HandTipRight = 23
        self.JointType_ThumbRight = 24


class JointConnections(JointType):
    r"""Enumerates all joint connections."""
    
    def __init__(self):
        super(JointConnections, self).__init__()
    
    def get_connections(self, num_joints=20):
        r"""Returns all joint connections.
        
        Args:
            num_joints (int): Number of joints in structure. It can be either
            20 (Kinect V1) or 25 (Kinect V2). Default is 20.
        
        Returns:
            A dictionary where each `key` is a connection identifier string and
            corresponding `value` is a tuple of two enumerated joint types that
            encodes a linear connection between them.
        
        """
        assert num_joints == 20 or num_joints == 25
        if num_joints == 20:
            connections = {
                'HD_NK': (self.JointType_Head, self.JointType_Neck),
                'NK_SM': (self.JointType_Neck, self.JointType_SpineMid),
                'SM_SB': (self.JointType_SpineMid, self.JointType_SpineBase),
                'NK_SL': (self.JointType_Neck, self.JointType_ShoulderLeft),
                'NK_SR': (self.JointType_Neck, self.JointType_ShoulderRight),
                'SB_HL': (self.JointType_SpineBase, self.JointType_HipLeft),
                'SB_HR': (self.JointType_SpineBase, self.JointType_HipRight),
                'SL_EL': (self.JointType_ShoulderLeft, self.JointType_ElbowLeft),
                'EL_WL': (self.JointType_ElbowLeft, self.JointType_WristLeft),
                'WL_HL': (self.JointType_WristLeft, self.JointType_HandLeft),
                'SR_ER': (self.JointType_ShoulderRight, self.JointType_ElbowRight),
                'ER_WR': (self.JointType_ElbowRight, self.JointType_WristRight),
                'WR_HR': (self.JointType_WristRight, self.JointType_HandRight),
                'HL_KL': (self.JointType_HipLeft, self.JointType_KneeLeft),
                'KL_AL': (self.JointType_KneeLeft, self.JointType_AnkleLeft),
                'AL_FL': (self.JointType_AnkleLeft, self.JointType_FootLeft),
                'HR_KR': (self.JointType_HipRight, self.JointType_KneeRight),
                'KR_AR': (self.JointType_KneeRight, self.JointType_AnkleRight),
                'AR_FR': (self.JointType_AnkleRight, self.JointType_FootRight)
            }
        else:
            connections = {
                'HD_NK': (self.JointType_Head, self.JointType_Neck),
                'NK_SS': (self.JointType_Neck, self.JointType_SpineShoulder),
                'SS_SM': (self.JointType_SpineShoulder, self.JointType_SpineMid),
                'SM_SB': (self.JointType_SpineMid, self.JointType_SpineBase),
                'SS_SL': (self.JointType_SpineShoulder, self.JointType_ShoulderLeft),
                'SS_SR': (self.JointType_SpineShoulder, self.JointType_ShoulderRight),
                'SB_HL': (self.JointType_SpineBase, self.JointType_HipLeft),
                'SB_HR': (self.JointType_SpineBase, self.JointType_HipRight),
                'SL_EL': (self.JointType_ShoulderLeft, self.JointType_ElbowLeft),
                'EL_WL': (self.JointType_ElbowLeft, self.JointType_WristLeft),
                'WL_HL': (self.JointType_WristLeft, self.JointType_HandLeft),
                'WL_TL': (self.JointType_WristLeft, self.JointType_ThumbLeft),
                'HL_HT': (self.JointType_HandLeft, self.JointType_HandTipLeft),
                'SR_ER': (self.JointType_ShoulderRight, self.JointType_ElbowRight),
                'ER_WR': (self.JointType_ElbowRight, self.JointType_WristRight),
                'WR_HR': (self.JointType_WristRight, self.JointType_HandRight),
                'WR_TR': (self.JointType_WristRight, self.JointType_ThumbRight),
                'HR_HT': (self.JointType_HandRight, self.JointType_HandTipRight),
                'HL_KL': (self.JointType_HipLeft, self.JointType_KneeLeft),
                'KL_AL': (self.JointType_KneeLeft, self.JointType_AnkleLeft),
                'AL_FL': (self.JointType_AnkleLeft, self.JointType_FootLeft),
                'HR_KR': (self.JointType_HipRight, self.JointType_KneeRight),
                'KR_AR': (self.JointType_KneeRight, self.JointType_AnkleRight),
                'AR_FR': (self.JointType_AnkleRight, self.JointType_FootRight)
            }
        return connections


class PointCloud3D(JointConnections):
    r"""Point Cloud in 3D."""
    
    def __init__(self, data=None):
        assert data is None or \
        (type(data) is ndarray and len(data.shape) == 2 and \
        (data.shape[1] == 60 or data.shape[1] == 75))
        
        super(PointCloud3D, self).__init__()
        self._data = data
        if self._data is None:
            self._num_frames = None
            self._num_joints = None
            self._connections = None
        else:
            self._num_frames = self._data.shape[0]
            self._num_joints = self._data.shape[1] // 3
            self._connections = self.get_connections(self._num_joints)
    
    def mount(self, data):
        assert type(data) is ndarray and len(data.shape) == 2 and \
        (data.shape[1] == 60 or data.shape[1] == 75)
        
        self._data = data
        self._num_frames = self._data.shape[0]
        self._num_joints = self._data.shape[1] // 3
        self._connections = self.get_connections(self._num_joints)
    
    def plot(self, title=None, draw_joints=True, draw_connections=True, fps=10,
             hide_plot=False, save_as=None, **kwargs):
        assert type(self._data) is ndarray and len(self._data.shape) == 2 and \
        self._data.shape[0] == self._num_frames and \
        self._data.shape[1] == self._num_joints * 3 and fps > 0
        
        figure = plt.figure()
        plot3D = self._init_plots(figure, title, draw_joints, draw_connections)
        anim3D = anim.FuncAnimation(figure, self._update_plots, self._num_frames, None,
                                    fargs=([plot3D]), save_count=None,
                                    interval=1000/fps, repeat_delay=100,
                                    repeat=True, blit=False)
        if not hide_plot:
            plt.show()
        if save_as:
            dirname = os.path.dirname(save_as)
            if dirname and not os.path.isdir(dirname):
                os.makedirs(dirname)
            anim3D.save(save_as, **kwargs)
    
    def _init_plots(self, fig, title=None, draw_joints=True, draw_connections=True):
        ax = Axes3D(fig)
        ax.set_xlim3d(-1, 1)
        ax.set_ylim3d(-1, 1)
        ax.set_zlim3d(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title('' if title is None else str(title))
        plots = dict()
        if draw_joints:
            plots['NODES'] = ax.plot([], [], [], 'o', color='red', alpha=0.5)
        if draw_connections:
            for name, connection in self._connections.items():
                if name in ['HD_NK', 'NK_SM', 'SM_SB',
                            'NK_SL', 'NK_SR', 'SB_HL', 'SB_HR',
                            'NK_SS', 'SS_SM', 'SS_SL', 'SS_SR']:
                    color = 'black'
                    alpha = 0.5
                elif name in ['SL_EL', 'EL_WL', 'WL_HL',
                              'SR_ER', 'ER_WR', 'WR_HR',
                              'WL_TL', 'HL_HT', 'WR_TR', 'HR_HT']:
                    color = 'magenta'
                    alpha = 0.5
                elif name in ['HL_KL', 'KL_AL', 'AL_FL',
                              'HR_KR', 'KR_AR', 'AR_FR']:
                    color = 'blue'
                    alpha = 0.5
                else:
                    color = 'red'
                    alpha = 1.0
                plots[name] = ax.plot([], [], [], linewidth=2, color=color, alpha=alpha)
        return plots
    
    def _update_plots(self, frame, plots):
        xs = self._data[:, 0::3]
        ys = self._data[:, 1::3]
        zs = self._data[:, 2::3]
        for name, plot in plots.items():
            if name.upper() == 'NODES':
                plot[0].set_data(xs[frame], zs[frame])
                plot[0].set_3d_properties(ys[frame])
            else:
                connection = self._connections[name]
                plot[0].set_data(xs[frame, connection], zs[frame, connection])
                plot[0].set_3d_properties(ys[frame, connection])
