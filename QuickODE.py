import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from matplotlib import rc


r_0 = 6
theta_0 = np.pi/4
phi_0 = 0
# Velocities
v_0 = 0
omega_theta_0 = 100
omega_phi_0 = .5
k = 10
m1 = 2
m2 = 2
r_e = 3
fric = .25
mu = m1*m2 / (m1 + m2) 
delta = .1
alpha = fric
beta = fric
gamma = fric
g = 1 # Gravity constant
c = 1 # Speed of Light
P = (m1 + m2) * g * c # Power is fixed to balance Gravity (F = P/c = m * g)
f_light = 0
torque_z = .5

def model(s, t):
        r, theta, phi, v, omega_theta, omega_phi = s
        dr = v
        dtheta = omega_theta
        dphi = omega_phi
        dv = r * omega_theta ** 2 + r * np.sin(theta) ** 2 * omega_phi ** 2 - 2 * k * (r - r_e) / mu - (delta / mu) * (r - r_e)**2 - alpha * abs(v) / mu
        domega_theta = np.sin(theta) * np.cos(theta) * omega_phi ** 2 - 2 * v * omega_theta / r - beta * abs(omega_theta) / (r ** 2 * mu)
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

plt.figure()
plt.plot(time, r_s)
plt.show()