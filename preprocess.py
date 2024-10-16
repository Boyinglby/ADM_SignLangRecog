import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from transforms import Transforms3D, Feature
import os
import glob
import numpy as np


class JointType(object):
    r"""Enumerates all joint types."""
    
    def __init__(self):
                
        self.JointType_Head = 0
        self.JointType_Neck = 1
        self.JointType_ShoulderLeft = 2
        self.JointType_ShoulderRight = 3
        self.JointType_ElbowLeft = 4
        self.JointType_ElbowRight = 5
        self.JointType_WristLeft = 6
        self.JointType_WristRight = 7
        self.JointType_HandLeft = 8
        self.JointType_HandRight = 9
        self.JointType_SpineMid = 10
        self.JointType_SpineBase = 11
        self.JointType_HipLeft = 12
        self.JointType_HipRight = 13
        self.JointType_KneeLeft = 14
        self.JointType_KneeRight = 15
        self.JointType_AnkleLeft = 16
        self.JointType_AnkleRight = 17
        self.JointType_FootLeft = 18
        self.JointType_FootRight = 19
        
class JointConnections(JointType):
    r"""Enumerates all joint connections."""
    
    def __init__(self):
        super(JointConnections, self).__init__()
    
    def get_connections(self, num_joints=20):
        """Returns all joint connections.
        
        Args:
            num_joints (int): Number of joints in structure. 
        
        """
        connections = [
            (1, 2), (2, 4), (4, 6), (6, 8), # left arm
            (1, 3), (3, 5), (5, 7),  (7, 9),  # right arm
            (11, 12), (12, 14), (14, 16), (16, 18),  # left leg
            (11, 13), (13, 15), (15, 17), (17, 19),  # right leg
            (0, 1), (1, 10), (10, 11)  # Spine
        ]

        return connections

class PointCloud3D(JointConnections):
    r"""Point Cloud in 3D."""
    
    def __init__(self, name=None, data=None):
        super(PointCloud3D, self).__init__()
        self._data = data
        self.name = name
        if self._data is None:
            self._num_frames = None
            self._num_joints = None
            self._connections = None
        else:
            self._num_frames = self._data.shape[0]
            self._num_joints = self._data.shape[1] // 3
            self._connections = self.get_connections(self._num_joints)
    

    def get_joints(self, frame_data):
        joints = []
        for i in range(0, len(frame_data), 3):
            joints.append((frame_data[i], frame_data[i+1], frame_data[i+2]))
        return joints
    
    def _initplot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        joints = self.get_joints(self._data[0])
        for joint in joints:
            self.scat = self.ax.scatter(joint[0], joint[2], joint[1], c='r', marker='o')
        for connection in self._connections:
            joint1 = joints[connection[0]]
            joint2 = joints[connection[1]]
            self.line = self.ax.plot([joint1[0], joint2[0]], [joint1[2], joint2[2]], [joint1[1], joint2[1]], 'b')

        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()  
        self.zlim = self.ax.get_zlim() 
    
    
    def update(self,frame):

        # update the scatter plot:
        joints = self.get_joints(self._data[frame])
        self.ax.clear()
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)
        for joint in joints:
            self.scat = self.ax.scatter(joint[0], joint[2], joint[1], c='r', marker='o')
        for connection in self._connections:
            joint1 = joints[connection[0]]
            joint2 = joints[connection[1]]
            self.line = self.ax.plot([joint1[0], joint2[0]], [joint1[2], joint2[2]], [joint1[1], joint2[1]], 'b')
    
    def plot_animation(self, filename="example.gif"):
        self._initplot()
        ani = animation.FuncAnimation(fig=self.fig, func=self.update, frames=40, interval=30)
        plt.show()
        ani.save(filename, writer="pillow")
        
    def transform(self):
        self._data = Transforms3D.transform(self)._data
    
    def add_cosine_feature(self):
        self._data = Feature.create_cosine(self)._data
    
    def add_velocity_feature(self):
        self._data = Feature.create_velocity(self)._data
        
    def high_std_feature(self):
        self._data = Feature.feature_select(self)._data
        
        
        

#------------------------------------------------------------------------
# Load the data from the text file
def load_file_to_dataframe(file_path):
    # Define column names for 20 points with 3D coordinates (p1_x, p1_y, p1_z, ..., p20_x, p20_y, p20_z)
    coordinates = ['x', 'y', 'z']
    column_names = [f'p{point}_{coord}' for point in range(1, 21) for coord in coordinates]
    
    # Load the file into a DataFrame
    df = pd.read_csv(file_path, delimiter=r'\s+', header=None, names=column_names)
    
    return df

# #---------------save processed data as 2700 .txt file in a folder--------------------
# root = '.\data'
# files = sorted(glob.glob(f'{root}/*.txt'))
# AllGestures = []

# for file in files:
#     df = load_file_to_dataframe(file)
#     array = df.values

#     Gesture = PointCloud3D(f"Guesture{os.path.basename(file)}",array)
#     Gesture.transform()
#     Gesture.high_std_feature()
#     Gesture.add_cosine_feature()
#     Gesture.add_velocity_feature()
#     AllGestures.append(Gesture)
#     #Save the array to a text file
#     np.savetxt(f'./processed_data/{os.path.basename(file)}', 
#                 Gesture._data, delimiter=" ", fmt='%.7f')

# #--------------discard velocity/frame features--------------------
# root = '.\data'
# files = sorted(glob.glob(f'{root}/*.txt'))
# AllGestures = []

# for file in files:
#     df = load_file_to_dataframe(file)
#     array = df.values

#     Gesture = PointCloud3D(f"Guesture{os.path.basename(file)}",array)
#     Gesture.transform()
#     Gesture.high_std_feature()
#     Gesture.add_cosine_feature()
#     # Gesture.add_velocity_feature()
#     AllGestures.append(Gesture)
#     #Save the array to a text file
#     np.savetxt(f'./own/24features/{os.path.basename(file)}', 
#                 Gesture._data, delimiter=" ", fmt='%.7f')
    
    
#--------------discard velocity/frame features and try with few data--------------------
root = './own/trydata'
files = sorted(glob.glob(f'{root}/*.txt'))
AllGestures = []

for file in files:
    df = load_file_to_dataframe(file)
    array = df.values

    Gesture = PointCloud3D(f"Guesture{os.path.basename(file)}",array)
    Gesture.transform()
    Gesture.high_std_feature()
    Gesture.add_cosine_feature()
    # Gesture.add_velocity_feature()
    AllGestures.append(Gesture)
    #Save the array to a text file
    np.savetxt(f'./own/24features_4class/{os.path.basename(file)}', 
                Gesture._data, delimiter=" ", fmt='%.7f')