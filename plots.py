import helpers as h
# from constants import *

from scipy.integrate import RK45 # NOTE: this is the Runge-Kutta 4(5) integrator, paper uses 4 th order, this should be more accurate
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pandas as pd
import scipy as sp


for period in tqdm.tqdm(np.linspace(3,6,6)):
    eq27 = h.DWDT(h.surge_sine,period)
    break
    lambda0 = 6.0
    w0 = eq27.U1/eq27.R*lambda0
    print(w0)
    print('starting integration...')
    times, omegas = eq27.solve(t0=0.,w0=w0, log=True)
    print('integration complete')
    
# times = times[times>3*h.PERIOD] # Filter out the first 3 periods
# omegas = omegas[times>3*h.PERIOD] # Filter out the first 3 periods
# read in data from log file to pandas dataframe
df_1 = pd.read_csv('log_t0_0.0_w0_80.0_period_1.0.txt', sep=',', header=None)
df_2 = pd.read_csv('log_t0_0.0_w0_80.0_period_2.0.txt', sep=',', header=None)
df_3 = pd.read_csv('log_t0_0.0_w0_80.0_period_3.0.txt', sep=',', header=None)
df_4 = pd.read_csv('log_t0_0.0_w0_80.0_period_4.0.txt', sep=',', header=None)
df_5 = pd.read_csv('log_t0_0.0_w0_80.0_period_5.0.txt', sep=',', header=None)
df_6 = pd.read_csv('log_t0_0.0_w0_80.0_period_6.0.txt', sep=',', header=None)

df_1.columns = ['time','omega']
df_2.columns = ['time','omega']
df_3.columns = ['time','omega']
df_4.columns = ['time','omega']
df_5.columns = ['time','omega']
df_6.columns = ['time','omega']


df_1 = df_1[df_1['time']>9]
df_2 = df_2[df_2['time']>9*2]
df_3 = df_3[df_3['time']>9*3]
df_4 = df_4[df_4['time']>9*4]
df_5 = df_5[df_5['time']>9*5]
df_6 = df_6[df_6['time']>9*6]

lambda0 = 6.0
norm = (eq27.U1/eq27.R*lambda0)
norm = 1
time_averaged = []
time_averaged.append(h.time_averaged(df_1['time'].to_numpy(),df_1['omega'].to_numpy())/norm)
time_averaged.append(h.time_averaged(df_2['time'].to_numpy(),df_2['omega'].to_numpy())/norm)
time_averaged.append(h.time_averaged(df_3['time'].to_numpy(),df_3['omega'].to_numpy())/norm)
time_averaged.append(h.time_averaged(df_4['time'].to_numpy(),df_4['omega'].to_numpy())/norm)
time_averaged.append(h.time_averaged(df_5['time'].to_numpy(),df_5['omega'].to_numpy())/norm)
time_averaged.append(h.time_averaged(df_6['time'].to_numpy(),df_6['omega'].to_numpy())/norm)
print(time_averaged)
# interpolate the values of time_averaged to get a smooth curve

periods = np.linspace(1,6,6)
times = np.linspace(1,6,60)
# time_averaged = np.interp(times,times,time_averaged)
cs = sp.interpolate.PchipInterpolator(periods,time_averaged)
xs = np.arange(-0.5, 9.6, 0.1)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(xs, cs(xs), label="S")
ax.set_xlim(-0.5, 9.5)
ax.legend(loc='lower left', ncol=2)
plt.show()

surge_vel_avg = periods*h.AMPLITUDE/eq27.U1
plt.plot(surge_vel_avg,time_averaged)
plt.xlabel('surge velocity (normalized)')
plt.ylabel('mean rotation rate (rad/s)')
# plt.ylim(0.9,1.)
# plt.title('Angular Velocity vs Time')
# plt.plot(times, eq27.surge_function(times,eq27.period))
# plt.xlabel('Time (s)')
# plt.ylabel('Surge Velocity (m/s)')
# plt.title('Surge Velocity vs Time')
plt.show()
