import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

import argparse
import math

matplotlib.use('MacOSX')

var = .1

# params have form [dx, dt]
### state calculations ###
def upwind_sim_1D(params, initial_state, u):
    #print("This is the 1D upwind simulation!")
    N = params[0] 
    nsteps = params[1]
    dx = params[2]
    dt = params[3]
    # make transport matrix
    transport_matrix = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        if u[i] >= 0:
            transport_matrix[i][(i-1)%N] = u[(i-1)%N]*dt/dx
            transport_matrix[i][i] = 1-u[i]*dt/dx
        else:
            transport_matrix[i][i] = 1 + u[i]*dt/dx
            transport_matrix[i][(i+1)%N] = -u[(i+1)%N]*dt/dx
    # check_stability(transport_matrix)
 
    # compute states
    states = [] 
    state = initial_state
    states.append(state)
    for i in range(nsteps):
        state = np.matmul(transport_matrix, state)
        states.append(state)
    return states
    
def shasta_sim_1D(params, initial_state, u):
    N = params[0] 
    nsteps = params[1]
    dx = params[2]
    dt = params[3]

    states = []
    states.append(initial_state)
    state = states[0]
    next_state = np.empty(state.shape)
    
    anti_dif = np.empty(state.shape)

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
    sys.stdout.write('\n')
    return states

def fpe_sim_1D(params, initial_state, u, d):
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
def make_gaussian(nodes):
    x = np.linspace(-1.0, +1.0, nodes)
    return .5*np.exp(-x ** 2/(2 * var * var))

def make_box(nodes):
    x = np.zeros(nodes)
    x[math.floor(3*nodes/8):math.floor(5*nodes/8)].fill(1)
    return x

def make_uniform(nodes, u):
    return np.full(nodes, u)

def make_quadratic(nodes):
    x = np.linspace(-1.0, +1.0, nodes)
    return -k * x 

def make_bistable(nodes, wind):
    x = np.linspace(-1.0, +1.0, nodes)
    u = -k*x*(1 - 4*x*x) + k*x*x*x 
    plt.plot(x, -k*(x*4)*x*(1 - (x*4)*x))
    plt.plot(x, u)
    return u

def make_2D_ex(nodes):
    l = np.linspace(-1.0, +1.0, nodes)
    (x, y) = np.meshgrid(l, l)
    return x, y, np.array(np.exp(- x**2 - y**2))


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
    start, end = 100, 2000 
    difs = np.empty(end-start)            
    for n in range(start, end):
        sys.stdout.write("\rChecking N = " + str(n))
        dx = .1 
        params = [n, 105, dx, dx] 
        states1 = sim_func(params, initial_func(params[0]), u_func(params[0], .1))

        params = [2*n, 105, dx/2, dx]
        states2 = sim_func(params, initial_func(params[0]), u_func(params[0], .1))

        dif_state = np.zeros(n)
        for i in range(n):
            dif_state[i] = states1[100][i] - states2[100][2*i]    
        difs[n - start] = np.sqrt(np.sum((dif_state ** 2)/n))
    return difs
         
        
### animations ###    
def animate_states_1D(states):
    fig, ax = plt.subplots()
    nodes = states[0].shape[0]
    x = np.linspace(-1.0, +1.0, nodes)
    line, = ax.plot(x, states[0])
    ax.set_ylim([0, 2])

    def animate(i):
        line.set_ydata(states[i])
        return line,

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(states)), init_func=init,
                                  interval=25, blit=False)
    plt.show()

'''
def animate_states_2D(states):
    # check out that stack exchange answer 
'''
    


### main ###
def main(arguments):
    parser = argparse.ArgumentParser()
    x, y, z = make_2D_ex(10)
    print(x.shape)
    print(y.shape)
    print(z.shape)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.show()



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
