import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import argparse
import math

dx = 1.0 
dt = 0.1 
nsteps = 800
N = 500
var = .1


### state calculations ###
def upwind_sim_1D(initial_state, u):
    print("This is the 1D upwind simulation!")
    N = initial_state.size

    # make transport matrix
    transport_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        if u[i] >= 0:
            transport_matrix[i][(i-1)%N] = u[(i-1)%N]*dt/dx
            transport_matrix[i][i] = 1-u[i]*dt/dx
        else:
            transport_matrix[i][i] = 1 + u[i]*dt/dx
            transport_matrix[i][(i+1)%N] = -u[(i+1)%N]*dt/dx
    check_stability(transport_matrix)
 
    # compute states
    states = [] 
    state = initial_state
    states.append(state)
    for i in range(nsteps):
        state = np.matmul(transport_matrix, state)
        states.append(state)
    return states
    
def shasta_sim_1D(initial_state, u):
    print("This is the 1D SHASTA simulation.")
    N = initial_state.size

    states = []
    states.append(initial_state)
    state = states[0]
    next_state = np.empty(state.shape)
    
    anti_dif = np.empty(state.shape)
    # build transport matrix
    transport = np.empty((N, N))
    for i in range(N):
        q_plus = (.5 - u[i]*dt/dx)/(1 + (u[(i+1)%N] - u[i])*dt/dx)
        q_minus = (.5 + u[i]*dt/dx)/(1 - (u[(i-1)%N] - u[i])*dt/dx)
        next_state[i] = .5 * q_minus * q_minus * (state[(i-1)%N] - state[i]) \
                      + .5 * q_plus * q_plus * (state[(i+1)%N] - state[i]) \
                      + (q_minus + q_plus)*state[i]
        transport[i][(i-1)%N] = .5 * q_minus * q_minus
        transport[i][i] = -.5 * q_minus * q_minus - .5 * q_plus * q_plus  
        transport[i][(i+1)%N] = .5 * q_plus * q_plus
    check_stability(transport)


    for i in range(nsteps):
        sys.stdout.write("\rCompute state " + str(i))
        # compute update
        for j in range(N):
            q_plus = (.5 - u[j]*dt/dx)/(1 + (u[(j+1)%N] - u[j])*dt/dx)
            q_minus = (.5 + u[j]*dt/dx)/(1 - (u[(j-1)%N] - u[j])*dt/dx)
            next_state[j] = .5 * q_minus * q_minus * (state[(j-1)%N] - state[j]) \
                          + .5 * q_plus * q_plus * (state[(j+1)%N] - state[j]) \
                          + (q_minus + q_plus)*state[j]

        # compute antidiffusion
        for j in range(N):
            delta_p1 = next_state[(j+1)%N] - next_state[j]
            delta_m1 = next_state[j] - next_state[(j-1)%N]
            delta_p3 = next_state[(j+2)%N] - next_state[(j+1)%N] 
            delta_m3 = next_state[(j-1)%N] - next_state[(j-2)%N]
            f_p = np.sign(delta_p1) * \
                        max(0, min(np.sign(delta_p1)*delta_m1, \
                        1/8*abs(delta_p1), \
                        np.sign(delta_p1)*delta_p3))
            f_m = np.sign(delta_m1) * \
                        max(0, min(np.sign(delta_m1)*delta_m3, \
                        1/8*abs(delta_m1), \
                        np.sign(delta_m1)*delta_p1))
            anti_dif[j] = - f_p + f_m


        next_state = np.add(next_state, anti_dif)
        state = np.copy(next_state)
        states.append(state)
        #print(state)
    sys.stdout.write('\n')
    return states

def fpe_sim_1D(initial_state, u, d):
    print("This is the 1D Fokker-Planck simulation.")
    N = initial_state.size

    states = []
    states.append(initial_state)
    state = states[0]
    next_state = np.empty(state.shape)
    
    for i in range(nsteps):
        # compute update    
        sys.stdout.write('\rCompute state ' + str(i))
        for j in range(N):
            next_state[j] = state[j]*(1 - 2*d[j]*dt/(dx*dx) - (u[(j+1)%N] - u[(j-1)%N])*dt/(2*dx)) \
                          + state[(j+1)%N]*(u[j]*dt/(2*dx) + (d[(j+1)%N] - d[(j-1)%N])*dt/(4*dx*dx) \
                              + d[j]*dt/(dx*dx)) \
                          + state[(j-1)%N]*(-u[j]*dt/(2*dx) - (d[(j+1)%N] - d[(j-1)%N])*dt/(4*dx*dx) \
                              + d[j]*dt/(dx*dx))

        state = np.copy(next_state)
        states.append(state)
    sys.stdout.write('\n')
    return states


### initial conditions ###
def make_gaussian():
    x = np.linspace(-1.0, +1.0, N)
    return .5*np.exp(-x ** 2/(2 * var * var))

def make_box():
    x = np.zeros(N)
    x[math.floor(3*N/8):math.floor(5*N/8)].fill(1)
    return x

k = 2
def make_quadratic():
    x = np.linspace(-1.0, +1.0, N)
    return -k * x 

def make_bistable():
    x = np.linspace(-1.0, +1.0, N)
    u = -k*x*(1 - 4*x*x) + k*x*x*x 
    plt.plot(x, -k*(x*4)*x*(1 - (x*4)*x))
    plt.plot(x, u)
    return u


### checker functions ###
def check_stability(transport):
    eig_values, _ = np.linalg.eig(transport)
    print("eigenvalues: " + str(eig_values))
    if np.absolute(eig_values).max() > 1:
        print("This scheme is unstable; the maximum eigenvalue magnitude is " \
            + str(np.absolute(eig_values).max()))
    else:
        print("This scheme is stable; the maximum eigenvalue magnitude is " \
            + str(np.absolute(eig_values).max()))

def check_convergence(sim_func, initial_func, u_func):
    for n in arange(10, 1000):
        N = n
        dx = 10/N 
        states1 = sim_func(initial_func(), u_func())
        N = 2*n
        dx = 10/N
        states2 = sim_func(initial_func(), u_func())
        
         
        


### animations ###    
def animate_states_1D(states):
    fig, ax = plt.subplots()
    	
    x = np.linspace(-1.0, +1.0, N)
    line, = ax.plot(x, states[0])
    ax.set_ylim([0, 2])

    def animate(i):
        line.set_ydata(states[i])
        return line,

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(1, nsteps), init_func=init,
                                  interval=25, blit=False)
    plt.show()
    

### main ###
def main(arguments):
    parser = argparse.ArgumentParser()
    
    print('Hello world!')
    wind = np.linspace(0, -10, N)
    wind.fill(5)
    #wind = np.exp(-wind*wind/(2*var*var))
    #states = shasta_sim_1D(make_box(), wind)
    states = upwind_sim_1D(make_box(), wind)
    u = make_bistable()
    #dif = np.linspace(-1.0, +1.0, N)
    #dif = np.exp(-dif*dif/(2*var*var))
    #dif.fill(1)
    #states = shasta_sim_1D(make_box(), u)
    animate_states_1D(states)
    


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
