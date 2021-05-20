import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import random as random
import time

# Isaiah Ertel Andre Suzanne
# Updated 11:30 a.m. 5/19/2021
# State Space for the Spinning dumbbell with spring connection

start = time.time()

# Loop to populate data file with multiple iterations of randomly choosen velocities at fixed energies

# Define our constants and starting values
# Energy
E = .5
# Positions
r_0 = 150
theta_0 = 0.1
phi_0 = 0.1
# Velocities
v_0 = 1
omega_theta_0 = 0.0001
omega_phi_0 = 0.0001
# Forces
D_theta = 1
# Constants
k = 1
mu = 1


# Defines our model using the State Space Flow Equations
def model(s, t):
    r, theta, phi, v, omega_theta, omega_phi = s
    dr = v
    dtheta = omega_theta
    dphi = omega_phi
    dv = r*omega_theta**2 + r*np.sin(theta)**2*omega_phi**2 - 2*k*r/mu
    domega_theta = np.sin(theta)*np.cos(theta)*omega_phi**2 - 2*v*omega_theta/r + D_theta/r
    domega_phi = -2*omega_theta*omega_phi/np.tan(theta) - 2*v*omega_phi/r
    return [dr, dtheta, dphi, dv, domega_theta, domega_phi]


# Solve ODE
s_0 = [r_0, theta_0, phi_0, v_0, omega_theta_0, omega_phi_0]
time = np.linspace(0, 100, 10000)
solution = integrate.odeint(model, s_0, time)

# Form Solutions
r = solution[:, 0]
theta = solution[:, 1]
phi = solution[:, 2]
v = solution[:, 3]
omega_theta = solution[:, 4]
omega_phi = solution[:, 5]

# Graph Projections of all Trajectories (ax1, ax2, ax3, ax4, ax5, ax6)
f, axs = plt.subplots(2, 3)
axs[0, 0].plot(time, r)
axs[0, 0].set_title('Radius')
axs[0, 1].plot(time, theta)
axs[0, 1].set_title('Theta')
axs[1, 0].plot(time, phi)
axs[1, 0].set_title('Phi')
axs[1, 1].plot(time, v)
axs[1, 1].set_title('Radial Velocity')
axs[0, 2].plot(time, omega_theta)
axs[0, 2].set_title('Omega Theta')
axs[1, 2].plot(time, omega_phi)
axs[1, 2].set_title('Omega Phi')


# Analysis
# Check Energy during the orbit
# Check for fixed points
def check_radial_fixed_point(velocity_solution):
    # Need system to slow down
    for v in velocity_solution:
        # TODO - This
        check_radial_fixed_threshold = .00001


def check_slope(variable1, variable2):
    check_slope_threshold = .00001
    # TODO - This also?!?!?!?!?!?!?!??!?!?!?!?!?!?!??!?!?!?!??!?!?!??!?!?!?!?!??!?!?!?HUHHHHHH?!?!?!??!?!?

# Show Plot after Loop is finished
plt.show()
