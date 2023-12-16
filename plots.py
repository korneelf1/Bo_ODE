import helpers as h
# from constants import *

from scipy.integrate import RK45 # NOTE: this is the Runge-Kutta 4(5) integrator, paper uses 4 th order, this should be more accurate
import matplotlib.pyplot as plt
import numpy as np
import tqdm


for period in tqdm.tqdm(np.linspace(2,6,10)):
    eq27 = h.DWDT(h.surge_sine,period)
    lambda0 = 6.0
    w0 = eq27.U1/eq27.R*lambda0

    print('starting integration...')
    times, omegas = eq27.solve(t0=0.,w0=w0, log=True)
    print('integration complete')
    break
# times = times[times>3*h.PERIOD] # Filter out the first 3 periods
# omegas = omegas[times>3*h.PERIOD] # Filter out the first 3 periods
print(h.time_averaged(times,omegas))


plt.subplot(2,1,1)
plt.plot(times,omegas)
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Angular Velocity vs Time')
plt.subplot(2,1,2)
plt.plot(times, eq27.surge_function(times,eq27.period))
plt.xlabel('Time (s)')
plt.ylabel('Surge Velocity (m/s)')
plt.title('Surge Velocity vs Time')
plt.show()
