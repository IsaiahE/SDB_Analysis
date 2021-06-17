import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Plots in 3 dimensions to more easily look for interesting behavior
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

csv_filename = 'AnimationData.csv'
path = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis\Energy_0.1_iniius_16_inieta_1.5_iniphi_0_iniity_0_inieta_0_iniphi_20_EM_rce_1_Sprant_1_Redass_1_Equius_1_Friant_0_'
positionData_df = pd.read_csv(path + '\\' + csv_filename)
positionData_df.pop('Unnamed: 0')

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

plt.plot(x, y, z)
plt.show()
