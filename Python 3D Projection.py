import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os

var_name_list = ['IR_1-20', 'ER_1-20', 'K_1-20', 'D_1-20', 'M1_1-20', 'IT_1-20']
csv_filename = 'AnimationData.csv'
path_to_folder_dir = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis'

for var_name in var_name_list:
    path_to_folder = path_to_folder_dir + '\\' + var_name
    for file in os.listdir(path_to_folder):

        try:
            positionData_df = pd.read_csv(path_to_folder + '\\' + file + '\\' + csv_filename)

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

        except Exception:
            print(file)
