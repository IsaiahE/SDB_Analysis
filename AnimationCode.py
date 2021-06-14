import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from pandas.io.parsers import read_csv
from matplotlib import animation

# Code Used for Animating
path = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis\Energy_0.1_iniius_16_inieta_1.5_iniphi_0_iniity_0_inieta_0_iniphi_20_EM_rce_1_Sprant_1_Redass_1_Equius_1_Friant_0_'
csv_filename = 'AnimationData.csv'
path2 = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis\Energy_0.1_iniius_16_inieta_1.5_iniphi_0_iniity_0_inieta_0_iniphi_20_EM_rce_1_Sprant_1_Redass_1_Equius_10_Friant_0_'

positionData_df = pd.read_csv(path + '\\' + csv_filename)
positionData_df.pop('Unnamed: 0')
positionData_df2 = pd.read_csv(path2 + '\\' + csv_filename)

r = positionData_df['r_s'].tolist()
theta = positionData_df['theta_s'].tolist()
phi = positionData_df['phi_s'].tolist()

x = []
y = []
z = []

for i in range(len(positionData_df['r_s'].tolist())):
    x.append(r[i] * np.sin(theta[i]) * np.sin(phi[i]))
    y.append(r[i] * np.sin(theta[i]) * np.cos(phi[i]))
    z.append(r[i] * np.cos(theta[i]))


x_t = np.array([])
for i in range(len(x)):
    x_t.append([x[i], y[i], z[i]])

"""
def init():
    line.set_data_3d([], [])    
    line.set_3d_properties([])
    return line,


def animate(i, line, x, y, z):
    x_s = []
    y_s = []
    z_s = []

    points = []

    for point in x[i]:
        x_s.append(point)
    for point in y[i]:
        y_s.append(point)
    for point in z[i]:
        z_s.append(point)

    line.set_data(x_s, y_s)
    line.set_3d_properties(z_s)
    return line,


start = [0, 1, 2]
goal = [10, 9, 6]
points = [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

fig = plt.figure()
ax = p3.Axes3D(fig)

ax.plot([x[0]], [y[0]], [z[0]], 'gs')
ax.plot([x[-1]],[y[-1]],[z[-1]], 'ms')
line, = ax.plot([], [], [], lw=2, color="red")

animation = FuncAnimation(fig, animate, 50, init_func=init, fargs=(line, x, y, z), interval=10000,   blit=True)

plt.show()
"""

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')

# choose a different color for each trajectory
colors = plt.cm.jet(np.linspace(0, 1, 2))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

print(lines)

# prepare the axes limits
ax.set_xlim((-250, 250))
ax.set_ylim((-250, 250))
ax.set_zlim((-250, 250))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data_3d([], [], [])
        # line.set_3d_properties([])

        pt.set_data_3d([], [], [])
        # pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=30, blit=True)

# Save as mp4. This requires mplayer or ffmpeg to be installed
#anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

plt.show()

"""
fig = plt.figure()
ax = fig.add_subplot(1,2,1,projection='3d')
ax.plot(x, y, z)

# now Lorentz
times = np.linspace(0, 4, 1000) 

# start_pts = 30. - 15.*np.random.random((20,3))  # 20 random xyz starting values


ax = fig.add_subplot(1,2,2,projection='3d')

ax.plot(x, y, z)

plt.show()
"""

"""
len_of_data = len(positionData_df)
fig, ax = plt.subplots(figsize=(5, 3))
ax = plt.axes(projection = '3d')
ax.set(xlim=(-3, 3), ylim=(-1, 1))
time = np.linspace(0, len_of_data)

dot = ax.scatter(positionData_df['r_s'].tolist()[0:100], positionData_df['theta_s'].tolist()[0:100], positionData_df['phi_s'].tolist()[0:100])


# plt.draw()
plt.show()
"""
"""
def animate(i):
    dot.set_data(positionData_df['r_s'].tolist()[i])


anim = FuncAnimation(fig, animate, interval=100, frames=len(time) - 1)

plt.draw()
plt.show()

anim.save(path + '\\' + 'AnimatedDumbbell.gif', writer='imagemagick')
"""
