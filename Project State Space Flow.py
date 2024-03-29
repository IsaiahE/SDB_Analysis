import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from matplotlib import rc
import random as random
import pandas as pd
import time
import os

# Isaiah Ertel Andre Suzanne
# Updated 7/15/2021
# State Space for the Spinning dumbbell with spring connection

start = time.time()

def main(constants):

    # Create Main Folder File
    path_to_folder = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis' + '\\' + constants['Main Folder Name']
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)

    # Create Info text document to store the dictionary
    if not os.path.isfile(path_to_folder + '//' + 'info.txt'):
        f = open(path_to_folder + '//' + 'info.txt', 'w')
        for key in list(constants.keys()):
            f.write(str(key) + ':  ' + str(constants[key]) + '\n')
            'Writing to Info.txt'
        f.close()

    # Define our constants and starting values
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
    delta = constants['Delta'] # Stiffining of the Spring
    # Forces
    D_theta = constants['D'] # Torque producing force
    alpha = fric
    beta = fric
    gamma = fric
    # Torque from Polarized Light    
    g = 1 # Gravity constant
    c = 1 # Speed of Light
    P = (m1 + m2) * g * c # Power is fixed to balance Gravity (F = P/c = m * g)
    f_light = constants['f']
    torque_z = 2 * P / f_light

    
    # D_theta function of theta probably
    
    # Defines our model using the State Space Flow Equations
    def model(s, t):
        r, theta, phi, v, omega_theta, omega_phi = s
        dr = v
        dtheta = omega_theta
        dphi = omega_phi
        dv = r * omega_theta ** 2 + r * np.sin(theta) ** 2 * omega_phi ** 2 - 2 * k * (r - r_e) / mu - (delta / mu) * (r - r_e)**2 - alpha * abs(v) / mu
        domega_theta = np.sin(theta) * np.cos(theta) * omega_phi ** 2 - 2 * v * omega_theta / r + D_theta / (r ** 2 * mu) - beta * abs(omega_theta) / (r ** 2 * mu)
        domega_phi = -2 * omega_theta * omega_phi * np.cos(theta) / np.sin(theta) - 2 * v * omega_phi / r - gamma * abs(omega_phi) / (r ** 2 * mu * np.sin(theta) ** 2) + torque_z / (mu * r ** 2 * np.sin(theta) ** 2)
        return [dr, dtheta, dphi, dv, domega_theta, domega_phi]

    # Solve ODE
    s_0 = [r_0, theta_0, phi_0, v_0, omega_theta_0, omega_phi_0]
    time = np.linspace(0, 5000, 100000)
    solution = integrate.odeint(model, s_0, time, rtol=1.49012e-10, atol=1.49012e-10)

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

        if constants['Settings'][0]:

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

        # Create a file for saving MatLab Plots
        if not os.path.exists(r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis' + '\\' + constants['Main Folder Name'] + '\\plots\\matlabplots'):
            os.makedirs(r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis' + '\\' + constants['Main Folder Name'] + '\\plots\\matlabplots')

        # Show Plot after Loop is finished
        save_plots = 'yes'
        if save_plots == 'yes':
            
            # Create a directory for which Variable is being iterated through
            path_to_folder = r'C:\Users\isaia\OneDrive - purdue.edu\Spinning Dumbbell Analysis' + '\\' + constants['Main Folder Name']
            IterationNumber = constants[constants['DataName']]
            if not os.path.exists(path_to_folder):
                os.makedirs(path_to_folder)

            # Create a directory to hold each set of plots
            path_to_plots_folder = path_to_folder + '\\' + 'plots'
            if not os.path.exists(path_to_plots_folder):
                os.makedirs(path_to_plots_folder)

            # Save info for Animation
            temp_dic = {'r_s': r_s[80000:], 'theta_s': theta_s[80000:], 'phi_s': phi_s[80000:], 
                        'velocity': v_s[80000:], 'omega_theta_s': omega_theta_s[80000:], 'omega_phi_s': omega_phi_s[80000:],
                        'time': time[80000:]}
            animation_data = pd.DataFrame(temp_dic)

            # Save Data to file for varying two variables
            if constants['Settings'][2]:

                animation_name = constants['DataName'] + '-' +  str(constants[constants['DataName']]) + '-' + str(constants['DataName2']) + str(constants[constants['DataName2']])
                print(animation_name)
                animation_data.to_csv(path_to_folder + '\\' + str(animation_name) + '.csv')
            
            # Save Data to file for varying one variable
            if not constants['Settings'][2]:

                print(constants['DataName'] + ': ' + str(IterationNumber))
                animation_data.to_csv(path_to_folder + '\\' + str(IterationNumber) + '.csv')
            
            # Permute through all possible Graphs (r, phi, theta, v, o_phi, o_theta)
            positions = [r_s, phi_s, theta_s]
            velocities = [v_s, omega_phi_s, omega_theta_s]
            positions_names = [r'$r$', r'$\phi$', r'$\theta$']
            velocities_names = [r'$v_r$', r'$\omega_\phi$', r'$\omega_\theta$']

            # Adjusting the font size of the plots
            if constants['Settings'][0]:

                fig_path = path_to_plots_folder + '\\' + str(IterationNumber)
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)

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
                plt.close()
            
            if constants['Settings'][1]:

                fig_path = path_to_plots_folder + '\\' + str(IterationNumber)
                if not os.path.exists(fig_path):
                    os.makedirs(fig_path)

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

                plt.close()

def run_parameter1D(Var_Name_Input, start, stop, steps=50):
    # Creates list of data with only one variable varried
    if not constants['Settings'][2]:
        constants['DataName'] = Var_Name_Input
        for i in range(0, steps + 1):
            constants[str(constants['DataName'])] = (i / steps) * stop + start * (steps - i) / steps
            main(constants)

def run_parameter2D():
    # Creates list of data with two variables varried
    if constants['Settings'][2]:
        constants['DataName'] = None
        constants['DataName2'] = None
        var_name = 'IOP'
        var_name2 = 'FR'
        start = 1
        stop = 100
        for j in range(start + 3, stop + 1):
            constants[str(var_name2)] = j / 10
            for i in range(start, stop + 1):
                constants[str(var_name)] = i / 10
                main(constants)


if __name__ == '__main__':
    constants = {'Main Folder Name': '', 'DataName': 'FR', 'DataName2': 'IOP',
                'IR': 6, 'IT': np.pi / 4, 'IP': 0, 
                'IV': 0, 'IOT': 0, 'IOP': 1, 
                'D': 0, 'K': 10, 'ER': 1, 'FR': .05, 'f': 100, 'Delta': .05,
                'M1': 2, 'M2': 2, 
                'Settings': [False,          False,               False]}
                          # [Plot Lissajous, Plot Radius and Phi, Saves Multiple Variables]

    # Name for the Data Directory 
    constants['Main Folder Name'] = ''
    
    # Note : May want to run with IOP != 0. MatLab simulation had errors for IOP = 0 from some kind of discontinuity

    # Loop through multiple parameters
    # 
    # Generally the format is 
    # Change name with constants['Main Folder Name]
    # Change any parameters that you want different from above
    # run_parameter1D
    # 
    amu = 1.6e-27
    order_of_f = 1e14
    angstrom = 1e-10
    approx_spring_const = .36
    
    #constants['M1'] = 200000 * amu
    #constants['M2'] = 200000 * amu
    #constants['IR'] = 6 * angstrom
    #constants['ER'] = 1 * angstrom
    #constants['K'] = 1 * approx_spring_const

    constants['Main Folder Name'] = 'Trash'
    run_parameter1D('FR', 0.0001, 5, steps=100)
    #constants['Main Folder Name'] = 'Delta2'
    #constants['K'] = 2
    #run_parameter1D('Delta', 0.001, 15, steps=100)
    #constants['Main Folder Name'] = 'Delta3'
    #constants['K'] = 3
    #run_parameter1D('Delta', 0.001, 15, steps=100)
    #constants['Main Folder Name'] = 'Delta4'
    #constants['K'] = 4
    #run_parameter1D('Delta', 0.001, 15, steps=100)
    #constants['Main Folder Name'] = 'Delta5'
    #constants['K'] = 5
    #run_parameter1D('Delta', 0.001, 15, steps=100)
    