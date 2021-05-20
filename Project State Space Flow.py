import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import random as random
import pandas
import time

# Isaiah Ertel Andre Suzanne
# Updated 11:30 a.m. 5/19/2021
# State Space for the Spinning dumbbell with spring connection

start = time.time()


def main(constants):
    # Loop to populate data file with multiple iterations of randomly choosen velocities at fixed energies

    # Define our constants and starting values
    # Energy
    E = constants['Energy']
    # Positions
    r_0 = constants['inital_radius']
    theta_0 = constants['initial_theta']
    phi_0 = constants['initial_phi']
    # Velocities
    v_0 = constants['initial_velocity']
    omega_theta_0 = constants['initial_omega_theta']
    omega_phi_0 = constants['initial_omega_phi']
    # Forces
    D_theta = constants['EM Force']
    # Constants
    k = constants['Coupling Constant']
    mu = constants['Reduced Mass']
    r_e = constants['Equilibrium Radius']  # Equalibrium Point


    # Defines our model using the State Space Flow Equations
    def model(s, t):
        r, theta, phi, v, omega_theta, omega_phi = s
        dr = v
        dtheta = omega_theta
        dphi = omega_phi
        dv = r * omega_theta ** 2 + r * np.sin(theta) ** 2 * omega_phi ** 2 - 2 * k * (r - r_e) / mu
        domega_theta = np.sin(theta) * np.cos(theta) * omega_phi ** 2 - 2 * v * omega_theta / r + D_theta / r
        domega_phi = -2 * omega_theta * omega_phi / np.tan(theta) - 2 * v * omega_phi / r
        return [dr, dtheta, dphi, dv, domega_theta, domega_phi]


    # Solve ODE
    s_0 = [r_0, theta_0, phi_0, v_0, omega_theta_0, omega_phi_0]
    time = np.linspace(0, 100, 10000)
    solution = integrate.odeint(model, s_0, time)

    # Form Solutions
    r_s = solution[:, 0]
    theta_s = solution[:, 1]
    phi_s = solution[:, 2]
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
        near_zero_df = pandas.DataFrame.from_dict(temp_dictionary, orient='index')
        near_zero_df = near_zero_df.transpose()
        return near_zero_df


    def check_slope(variable1, variable2):
        check_slope_threshold = .00001
        # TODO - This also?!?!?!?!?!?!?!??!?!?!?!?!?!?!??!?!?!?!??!?!?!??!?!?!?!?!??!?!?!?HUHHHHHH?!?!?!??!?!?


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
        f, axs = plt.subplots(2, 3)
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


        # Show Plot after Loop is finished
        plt.show()


# TODO - Find the sweet spot for the initial values and constants below. This will be where the interesting stuff 
#  happens most often. Then we can itterate around these values more precisesly.
if __name__ == '__main__':
    constants = {'Energy': .1, 'inital_radius': .1, 'initial_theta': .1, 'initial_phi': .1, 'initial_velocity': .1,
                 'initial_omega_theta': .1, 'initial_omega_phi': .1, 'EM Force': .1, 'Coupling Constant': .1,
                 'Reduced Mass': .1, 'Equilibrium Radius': .1}

    for radius in range(1, 150):
        constants['inital_radius'] = radius
        main(constants)
