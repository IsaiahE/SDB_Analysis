import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from matplotlib import rc
import random as random
import pandas as pd
import time
import os

# Isaiah Ertel Andre Suzanne
# Updated 3:00 p.m. 7/1/2021
# State Space for the Spinning dumbbell with spring connection

start = time.time()

def main(constants):
    # Loop to populate data file with multiple iterations of randomly choosen velocities at fixed energies

    # Define our constants and starting values
    # Energy
    # E = constants['Energy'] # Initial Energy (Not currently used)
    # Positions
    r_0 = constants['IR'] # Initial Radius
    theta_0 = constants['IT'] # Initial Theta
    phi_0 = constants['IP'] # Initial Phi
    # Velocities
    v_0 = constants['IV'] # Initial Angular Velocity in Phi
    omega_theta_0 = constants['IOT'] # Initial Angular Velocity in Phi
    omega_phi_0 = constants['IOP'] # Initial Angular Velocity in Phi
    # Constants
    k = constants['K'] # Spring Constant k
    m1 = constants['M1'] # Mass of ball 1
    m2 = constants['M2'] # Mass of ball 2
    r_e = constants['ER']  # Equalibrium Point
    fric = constants['FR'] # Friction Constant
    mu = m1*m2 / (m1 + m2) # Reduced Mass
    # Forces
    D_theta = constants['D'] # Torque producing force
    alpha = fric
    beta = fric
    gamma = fric
    torque_z = 1

    def velocity_friction1(friction_const, v):
        return friction_const * v ** 2

    def velocity_friction2(friction_const, o_t, r):
        return friction_const * o_t ** 2 * r ** 3

    def velocity_friction3(friction_const, o_p, r):
        return friction_const * o_p ** 2 * r ** 3

    # D_theta function of theta probably
    
    # Defines our model using the State Space Flow Equations
    def model(s, t):
        r, theta, phi, v, omega_theta, omega_phi = s
        dr = v
        dtheta = omega_theta
        dphi = omega_phi
        dv = r * omega_theta ** 2 + r * np.sin(theta) ** 2 * omega_phi ** 2 - 2 * k * (r - r_e) / mu - alpha * abs(v) / mu
        domega_theta = np.sin(theta) * np.cos(theta) * omega_phi ** 2 - 2 * v * omega_theta / r + D_theta / (r ** 2 * mu) - beta * abs(omega_theta) / (r ** 2 * mu)
        domega_phi = -2 * omega_theta * omega_phi * np.cos(theta) / np.sin(theta) - 2 * v * omega_phi / r - gamma * abs(omega_phi) / (r ** 2 * mu) + torque_z / (mu * r ** 2 * np.sin(theta) ** 2)
        return [dr, dtheta, dphi, dv, domega_theta, domega_phi]

    # Solve ODE
    s_0 = [r_0, theta_0, phi_0, v_0, omega_theta_0, omega_phi_0]
    time = np.linspace(0, 1000, 100000)
    solution = integrate.odeint(model, s_0, time)

    # Form Solutions
    r_s = solution[:, 0]
    theta_s = solution[:, 1]
    phi_s = np.mod(solution[:, 2], 2 * np.pi)
    v_s = solution[:, 3]
    omega_theta_s = solution[:, 4]
    omega_phi_s = solution[:, 5]


    # Analysis
    # Check Energy during the orbit
    # Check for fixed points
    def check_fixed_point(total_solution):
        # Need system to slow down
        check_radial_fixed_threshold = .00001
        check_theta_fixed_threshold = .00001
        check_phi_fixed_threshold = .00001
        temp_dictionary = {'TimeR': [], 'Position': [], 'Velocity': [],
                           'TimeT': [], 'Theta': [], 'Omega_theta': [],
                           'TimeP': [], 'Phi': [], 'Omega_Phi': []}
        for i in range(len(total_solution[:, 3])):
            # Radial Fixed Point
            if -check_radial_fixed_threshold < total_solution[:, 3][i] < check_radial_fixed_threshold:
                temp_dictionary['TimeR'].append(time[i])
                temp_dictionary['Position'].append((total_solution[:, 0][i]))
                temp_dictionary['Velocity'].append(total_solution[:, 3][i])
        for i in range(len(total_solution[:, 3])):
            # Theta Fixed Point
            if -check_theta_fixed_threshold < total_solution[:, 4][i] < check_theta_fixed_threshold:
                temp_dictionary['TimeT'].append(time[i])
                temp_dictionary['Theta'].append((total_solution[:, 1][i]))
                temp_dictionary['Omega_theta'].append(total_solution[:, 4][i])
        for i in range(len(total_solution[:, 3])):
            # Phi Fixed Point
            if -check_phi_fixed_threshold < total_solution[:, 5][i] < check_phi_fixed_threshold:
                temp_dictionary['TimeP'].append(time[i])
                temp_dictionary['Phi'].append((total_solution[:, 2][i]))
                temp_dictionary['Omega_Phi'].append(total_solution[:, 5][i])

        # Creates and returns a Pandas DataFrame to make data manipulation easier later
        near_zero_df = pd.DataFrame.from_dict(temp_dictionary, orient='index')
        near_zero_df = near_zero_df.transpose()
        return near_zero_df

    # Currently not being used
    def check_slope(variable1, variable2):
        check_slope_threshold = .1
        derivative = np.diff(variable2) / np.diff(variable1)
        validSlope = False
        count = 0

        for item in derivative:
            if 1 - check_slope_threshold < item < 1 + check_slope_threshold:
                count += 1

            else:
                count = 0

            if count >= 10000:
                validSlope = True
                break

        return validSlope

    # The Graphing will be performed only if a specific behavior from functions below
    plot_solutions = False
    fixed_points_df = check_fixed_point(solution)
    if fixed_points_df.count()['TimeR'] != 0 and fixed_points_df.count()['TimeT'] != 0:
        plot_solutions = True
    if fixed_points_df.count()['TimeR'] != 0 and fixed_points_df.count()['TimeP'] != 0:
        plot_solutions = True
    if fixed_points_df.count()['TimeT'] != 0 and fixed_points_df.count()['TimeP'] != 0:
        plot_solutions = True

    if plot_solutions:
        # Graph Projections of all Trajectories (ax1, ax2, ax3, ax4, ax5, ax6)
        f, axs = plt.subplots(4, 3)
        axs[0, 0].plot(time, r_s, 's', markersize=1)
        axs[0, 0].set_title('Radius')
        axs[0, 1].plot(time, theta_s, 's', markersize=1)
        axs[0, 1].set_title('Theta')
        axs[0, 2].plot(time, phi_s, 's', markersize=1)
        axs[0, 2].set_title('Phi')
        axs[1, 0].plot(time, v_s, 's', markersize=1)
        axs[1, 0].set_title('Radial Velocity')
        axs[1, 1].plot(time, omega_theta_s, 's', markersize=1)
        axs[1, 1].set_title('Omega Theta')
        axs[1, 2].plot(time, omega_phi_s, 's', markersize=1)
        axs[1, 2].set_title('Omega Phi')
        axs[2, 0].plot(v_s, omega_theta_s, 's', markersize=1)
        axs[2, 0].set_title('Velocity vs Omega_Theta')
        axs[2, 1].plot(v_s, omega_phi_s, 's', markersize=1)
        axs[2, 1].set_title('Velocity vs Omega_Phi')
        axs[2, 2].plot(omega_phi_s, omega_theta_s, 's', markersize=1)
        axs[2, 2].set_title('Omega_Phi vs Omega_Theta')
        axs[3, 0].plot(r_s, theta_s, 's', markersize=1)
        axs[3, 0].set_title('Radius vs Theta')
        axs[3, 1].plot(r_s, phi_s, 's', markersize=1)
        axs[3, 1].set_title('Radius vs Phi')
        axs[3, 2].plot(phi_s, theta_s, 's', markersize=1)
        axs[3, 2].set_title('Phi vs Theta')

        stringy = ''
        for key in constants.keys():
            if key == 'Change Var':
                continue
            if key == 'Direct Name':
                continue
            else:
                stringy += str(key) + '-' + str(constants[key]) + '_'
        stringy = stringy[:-1]

        # Show Plot after Loop is finished
        save_plots = 'yes'
        if save_plots == 'yes':
            
            # Create a directory for which Variable is being iterated through
            path_to_folder = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis' + '\\' + constants['Main Folder Name']
            IterationNumber = constants[constants['DataName']]
            print(constants['DataName'] + ': ' + str(IterationNumber))
            if not os.path.exists(path_to_folder):
                os.makedirs(path_to_folder)

            # Create a directory to hold each set of plots
            path_to_plots_folder = path_to_folder + '\\' + 'plots'
            if not os.path.exists(path_to_plots_folder):
                os.makedirs(path_to_plots_folder)
            fig_path = path_to_plots_folder + '\\' + str(IterationNumber)
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)

            # Save info for Animation
            temp_dic = {'r_s': r_s, 'theta_s': theta_s, 'phi_s': phi_s, 'velocity': v_s, 'omega_theta_s': omega_theta_s, 'omega_phi_s': omega_phi_s,'time': time}
            animation_data = pd.DataFrame(temp_dic)
            animation_data.to_csv(path_to_folder + '\\' + str(IterationNumber) + '.csv')

            # Permute through all possible Graphs (r, phi, theta, v, o_phi, o_theta)
            positions = [r_s, phi_s, theta_s]
            velocities = [v_s, omega_phi_s, omega_theta_s]
            positions_names = [r'$r$', r'$\phi$', r'$\theta$']
            velocities_names = [r'$v_r$', r'$\omega_\phi$', r'$\omega_\theta$']

            # Adjusting the font size of the plots
            if constants['Settings'][0]:
                parameters = {'axes.labelsize': 18, 'axes.titlesize': 25}
                plt.rcParams.update(parameters)
                plt.figure()
                for index1 in [0, 1, 2]:
                    for index2 in [0, 1, 2]:
                        if index1 < index2:
                            # Position vs Position Plots
                            plt.plot(positions[index1], positions[index2], 's', markersize=1)
                            plt.title(positions_names[index1] + '  vs  ' + positions_names[index2])
                            plt.xlabel(positions_names[index1])
                            plt.ylabel(positions_names[index2])
                            plt.savefig(fig_path + '\\positions' + str(index1) + str(index2) + '.png', dpi = 300)
                            plt.clf()

                            # Velocity vs Velocity Plots
                            plt.plot(velocities[index1], velocities[index2], 's', markersize=1)
                            plt.title(velocities_names[index1] + '  vs  ' + velocities_names[index2])
                            plt.xlabel(velocities_names[index1])
                            plt.ylabel(velocities_names[index2])
                            plt.savefig(fig_path + '\\velocities' + str(index1) + str(index2) + '.png', dpi = 300)
                            plt.clf()

                        if index1 == index2:
                            # Position vs Velocity Plots
                            plt.plot(positions[index1], velocities[index2], 's', markersize=1)
                            plt.title(positions_names[index1] + '  vs  ' + velocities_names[index2])
                            plt.xlabel(positions_names[index1])
                            plt.ylabel(velocities_names[index2])
                            plt.savefig(fig_path + '\\statespaces' + str(index1) + str(index2) + '.png', dpi = 300)
                            plt.clf()
            
            if constants['Settings'][1]:
                parameters = {'axes.labelsize': 18, 'axes.titlesize': 25}
                plt.rcParams.update(parameters)
                plt.figure()

                plt.plot(time, r_s, 's', markersize=1)
                plt.plot(time, phi_s, 's', markersize=1)
                plt.title('Radius and Phi vs Time')
                plt.xlabel('Time')
                plt.ylabel('Value')
                plt.savefig(fig_path + '\\TimePlotsRadPhi' + '.png', dpi = 300)
                plt.clf()

                plt.plot(phi_s, r_s, 's', markersize=1)
                plt.title('Radius vs Phi')
                plt.xlabel('Phi')
                plt.ylabel('Radius')
                plt.savefig(fig_path + '\\RadvsPhi' + '.png', dpi = 300)
                plt.clf()

if __name__ == '__main__':
    constants = {'Main Folder Name': 'Varying_Initial_Theta', 'DataName': 'IT',
                'IR': 2, 'IT': 1.5, 'IP': 0, 
                'IV': 0, 'IOT': 0, 'IOP': 1, 
                'D': 1, 'K': 2, 'ER': 1, 'FR': .01, 'TZ': 1,
                'M1': 2, 'M2': 2,
                'Settings': [False,          False]}
                          # [Plot Lissajous, Plot Radius and Phi]

    var_name_list = ['IT']
    for var_name in var_name_list:
        start = 0
        stop = 100
        for i in range(start, stop + 1):
            constants[str(var_name)] = np.pi * i/200
            main(constants)
        constants[var_name] = start
