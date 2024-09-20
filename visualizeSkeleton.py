import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the data from the text file
data = []
with open('bye_mahendra_1.txt', 'r') as file:
    for line in file:
        data.append([float(x) for x in line.split()])

# Assuming each frame is on a new line and structured as [x1, y1, z1, x2, y2, z2, ..., x20, y20, z20]
def get_joints(frame_data):
    joints = []
    for i in range(0, len(frame_data), 3):
        joints.append((frame_data[i], frame_data[i+1], frame_data[i+2]))
    return joints

# Define the connections between joints to form the skeleton
connections = [
    (1, 2), (2, 4), (4, 6), (6, 8), # left arm
    (1, 3), (3, 5), (5, 7),  (7, 9),  # right arm
    (11, 12), (12, 14), (14, 16), (16, 18),  # left leg
    (11, 13), (13, 15), (15, 17), (17, 19),  # right leg
    (0, 1), (1, 10), (10, 11)  # Spine
]

# Plot the skeleton
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

joints = get_joints(data[0])
for joint in joints:
    scat = ax.scatter(joint[0], joint[2], joint[1], c='r', marker='o')
for connection in connections:
    joint1 = joints[connection[0]]
    joint2 = joints[connection[1]]
    line = ax.plot([joint1[0], joint2[0]], [joint1[2], joint2[2]], [joint1[1], joint2[1]], 'b')

xlim = ax.get_xlim()
ylim = ax.get_ylim()  
zlim = ax.get_zlim()     
def update(frame):

    # update the scatter plot:
    joints = get_joints(data[frame])
    ax.clear()
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    for joint in joints:
        scat = ax.scatter(joint[0], joint[2], joint[1], c='r', marker='o')
    for connection in connections:
        joint1 = joints[connection[0]]
        joint2 = joints[connection[1]]
        line = ax.plot([joint1[0], joint2[0]], [joint1[2], joint2[2]], [joint1[1], joint2[1]], 'b')
    
    return (scat, line)    


ani = animation.FuncAnimation(fig=fig, func=update, frames=40, interval=30)
plt.show()
ani.save(filename="bye_example.gif", writer="pillow")






