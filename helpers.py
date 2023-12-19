import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK23,RK45 # NOTE: this is the Runge-Kutta 4(5) integrator, paper uses 4 th order, this should be more accurate
import tqdm
import numpy as np
# surge velocity functions
AMPLITUDE = 0.3
# PERIOD    = 1 # paper states between 1 and 6 sec
def surge_sine(t,period):
    return AMPLITUDE*np.sin(2*np.pi*t/period)

def surge_square(t,period):
    return AMPLITUDE*(t<period/2) - AMPLITUDE*(t>period/2)


def plot(function,period):
    t = np.linspace(0,period,100)
    plt.plot(t,function(t))
    plt.xlabel('Time (s)')
    plt.ylabel('Surge Velocity (m/s)')
    plt.title('Surge Velocity Plotter Output')
    plt.show()
    return 0 

# turbine power curve, values from page 13 under equation 3.1
C1 = 16.784
C2 = -1.510
C3 = 1.702
C4 = 8.764   
def cp0(lam):
    '''
    Returns the power coefficient for a given tip speed ratio
    see equation 3.1
    '''
    return (C1/(lam+C2)-C3)*np.exp(-C4/(lam+C2))

# paper mentions at page 13:
# To obtain time-resolved predictions of the 
# turbine rotation rate, torque, and power, 
# Equation 2.7 was numerically integrated over ten surge periods
# using a fourth-order Runge-Kutta scheme. 
# Timesteps were kept no larger than 0.001𝑇 
# to maintain numerical stability and accuracy. 
# The steady-flow turbine rotation rate 𝜔0 was used 
# as the initial condition, 
# and convergence was typically established 
# within a few forcing periods. 
# The model predictions for the amplitude, 
# phase, and time-average of each quantity 
# were computed from the final period in the 
# simulation.

class DWDT:
    '''
    Based on equation 2.7, page 5
    This class defines all constants for the equation and provides a method to solve it
    during initialization, the surge function is passed in as a parameter
    choose one of the surge functions defined above or create your own
    In refactoring, these variables should become global constants
    '''
    def __init__(self, surge_function, period):
        self.J = 0.0266 # see page 12
        self.K0 = 0.119
        self.K1 = 0.0112
        self.K2 = 6.96e-4
        self.RHO = 1.225
        self.PI = np.pi
        self.R = 1.17/2 # ipv 63 
        self.U1 = 7.8 # 11.4 
        self.dt_max = period/1e3 # paper states 0.001T
        self.t_max = 10*period
        self.period = period
        self.surge_function = surge_function

    def lam(self,t, w):
        '''
        Computes the tip-speed ratio based on Equation 2.4
        Args:
            t: time
            w: angular velocity
        '''
        return self.R*w/(self.U1-self.surge_function(t,self.period))
    
    def compute_dwdt(self, t,w):
        '''
        Based on equation 2.7, page 5

        Args:
            t: time
            w: angular velocity
        Returns:
            derivative of the angular velocity

        '''
        # print('in compute:' ,w)
        factor = 1/self.J+self.K2
        term1 = -self.K0
        term2 = -self.K1*w
        term3 = (0.5*self.RHO*np.pi*self.R**2)*(self.U1-self.surge_function(t,self.period))**3/w*cp0(self.lam(t,w))

        return factor*(term1+term2+term3)
                         

    def solve(self,t0,w0, log=False):
        '''
        Solves the differential equation
        Args:
            t0: initial time
            w0: initial angular velocity
        Returns:
            t: time array
            w: angular velocity array
        '''
        # print(self.t_max)
        # set up the ODE solver using parameters from the paper:
        solver = RK45(self.compute_dwdt,t0,[w0],t_bound = self.t_max,max_step=self.dt_max,atol=1e-5)
        # initialize objects to store the output
        output_omega = []
        output_t = []
        #  Use tqdm as a progress bar
        # with tqdm.tqdm(total=int(self.t_max / self.dt_max), desc="Integration Approximate Progress", unit='steps') as progress_bar:
        # run until integration is complete
        while solver.status == 'running':
            # take an integration step
            # print('step size:',solver.step_size)
            # print(solver.t)
            solver.step()
            output_t.append(solver.t)
            output_omega.append(solver.y[0])
            # print('from solver:', solver.y)
                # progress_bar.update(1)

        # if log write to txt file
        if log:
            with open(f'log_t0_{t0}_w0_{w0}_period_{self.period}.txt','w+') as f:
                for i in range(len(output_t)):
                    f.write(str(output_t[i]) + ',' + str(output_omega[i]) + '\n')
        # return the output
        return np.array(output_t), np.array(output_omega)
    
def time_averaged(times, values):
    '''
    Computes the time averaged value of a function
    Args:
        times: time array
        values: function values array
    Returns:
        time averaged value
    '''
    # return np.mean(values)
    return np.trapz(values,times)/(times[-1]-times[0])


if __name__ == '__main__':
    pass