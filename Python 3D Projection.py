import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

path = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis\Energy_0.1_iniius_16_inieta_1.5_iniphi_0_iniity_0_inieta_0_iniphi_20_EM_rce_1_Sprant_1_Redass_1_Equius_1_Friant_0_\AnimationData.csv'

path = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis\Energy_0.1_iniius_16_inieta_1.5_iniphi_0_iniity_0_inieta_0_iniphi_20_EM_rce_1_Sprant_1_Redass_1_Equius_1_Friant_0_'
path = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis\Energy_0.1_iniius_16_inieta_1.5_iniphi_0_iniity_0_inieta_0_iniphi_20_EM_rce_1_Sprant_1_Redass_1_Equius_1_Friant_0_'

csv_filename = 'AnimationData.csv'
positionData_df = pd.read_csv(path + '\\' + csv_filename)

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



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x,y,z)
plt.show()



