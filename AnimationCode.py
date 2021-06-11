import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

# Code Used for Animating
path = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis\Energy_0.1_iniius_16_inieta_1.5_iniphi_0_iniity_0_inieta_0_iniphi_20_EM_rce_1_Sprant_1_Redass_1_Equius_1_Friant_0_'
csv_filename = 'AnimationData.csv'

positionData_df = pd.read_csv(path + '\\' + csv_filename)
positionData_df.pop('Unnamed: 0')
print(positionData_df)


len_of_data = len(positionData_df)
fig, ax = plt.subplots(figsize=(5, 3))
ax = plt.axes(projection = '3d')
ax.set(xlim=(-3, 3), ylim=(-1, 1))
time = np.linspace(0, len_of_data)

dot = ax.scatter(positionData_df['r_s'].tolist()[0:100], positionData_df['theta_s'].tolist()[0:100], positionData_df['phi_s'].tolist()[0:100])


# plt.draw()
plt.show()

"""
def animate(i):
    dot.set_data(positionData_df['r_s'].tolist()[i])


anim = FuncAnimation(fig, animate, interval=100, frames=len(time) - 1)

plt.draw()
plt.show()

anim.save(path + '\\' + 'AnimatedDumbbell.gif', writer='imagemagick')
"""
