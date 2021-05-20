# SDB_Analysis (Spinning Dumbbell Analysis)
Code for analyzing the spinning dumbbell with a spring attachment with a constant torque from an electromagnetic field. The code allows for dissipative forces in order to find synchronization. 
The code uses scipy.integrate's ode solver to solve teh State Space Flow of the system 
The Flow is 6 Dimensional after solving in center of mass coordinates. 
Because a Poincare Section reduces the dimensionality by 1 and energy conservation allows us to another
dimension we can only bring the dimensionality down to 4. This makes it hard to visualy find any syncronization.
Instead we created algorithms to try and detect where syncronization might occur for given parameters.

Coding was done by Andre Suzanne and Isaiah Ertel
