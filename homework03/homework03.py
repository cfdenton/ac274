import sys
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import cm
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('MacOSX')

### Simulation functions ###
    # params:
    #     nsteps - number of simulation steps
    #     N - grid size
    #     initial - initial state (numpy array with dimension /N/)
    #     dt - time increment 
    #     dx - space increment

def diffusion_1D(params):
    N, dt, dx, nu, nsteps = params['N'], params['dt'], params['dx'], \
        params['nu'], params['nsteps']
    states = [params['initial']]
    
    for i in range(1, nsteps):
        states.append(np.empty(N))
        for j in range(N):
            states[i][j] = states[i-1][j] + \
                dt*nu/(dx*dx)*(states[i-1][(j+1)%N] - 2*states[i-1][j] + states[i-1][j-1])
    return states


def burgers_1D(params):
    N, dt, dx, nu, nsteps = params['N'], params['dt'], params['dx'], \
        params['nu'], params['nsteps']
    states = [params['initial']]    
    
    for i in range(1, nsteps):
        states.append(np.empty(N))
        for j in range(N):
            # compute next state
            states[i][j] = states[i-1][j]*(1 - \
                dt/(2*dx)*(states[i-1][(j+1)%N] - states[i-1][(j-1)%N])) \
                + nu*dt/(dx*dx)*(states[i-1][(j+1)%N] - 2*states[i-1][j] \
                + states[i-1][(j-1)%N]) 
    return states

def kpz_1D(params):
    N, dt, dx, nu, nsteps, rand, lamb = params['N'], params['dt'], params['dx'], \
        params['nu'], params['nsteps'], params['rand'], params['lambda']
    states = [params['initial']]

    for i in range(1, nsteps):
        states.append(np.empty(N))
        noise = np.random.randn(N)
        for j in range(N):
            # compute next state
            states[i][j] = states[i-1][j] + dt*lamb/(8*dx*dx)*((states[i-1][(j+1)%N] - \
                states[i-1][(j-1)%N]) ** 2) \
                + dt*nu/(dx*dx)*(states[i-1][(j+1)%N] - 2*states[i-1][j] + states[i-1][(j-1)%N]) \
                + dt*rand*noise[j]
    return states            

def kpz_2D(params):
    N, dt, dx, nu, nsteps, rand, lamb = params['N'], params['dt'], params['dx'], \
        params['nu'], params['nsteps'], params['rand'], params['lambda']
    states = [params['initial']]

    for i in range(1, nsteps):
        sys.stdout.write('\rComputing state: ' + str(i))
        states.append(np.empty((N, N)))
        noise = np.random.randn(N, N)
        for j in range(N):
            for k in range(N):
                states[i][j][k] = states[i-1][j][k] + \
                    dt*lamb/(8*dx*dx)*((states[i-1][(j+1)%N][k] - states[i-1][(j-1)%N][k] \
                    + states[i-1][j][(k+1)%N] - states[i-1][j][(k-1)%N]) ** 2) \
                    + dt*nu/(dx*dx)*(states[i-1][(j+1)%N][k] + states[i-1][(j-1)%N][k] \
                    + states[i-1][j][(k+1)%N] + states[i-1][j][(k-1)%N] - 4*states[i-1][j][k]) \
                    + dt*rand*noise[j][k]
    sys.stdout.write('\n')
    return states        


def sample_2D():
    l = np.linspace(-1.0, +1.0, 10)
    (x, y) = np.meshgrid(l, l)
    z = np.exp(-x**2 - y**2)
    states = []
    for i in range(100):
        states.append(z + .01*i)
    return states

### Initial Functions ###
def make_gaussian_1D(nodes):
    return np.exp(-(np.linspace(-1.0, +1.0, nodes)**2)*10)

def make_uniform_1D(nodes):
    return np.full(nodes, .5)

def make_gaussian_2D(nodes):
    l = np.linspace(-1.0, +1.0, nodes)
    (x, y) = np.meshgrid(l, l)
    return np.exp(-x**2 -y**2)

def make_uniform_2D(nodes):
    return np.full((nodes, nodes), .5) 


### Check Functions ###
def convergence_1D(sim_func, initial_func, params):
    rms_val = []
    lower, upper = 10, 1000
    for i in range(lower, upper):
        sys.stdout.write('\rChecking convergence for N = ' + str(i))
        params['N'] = i
        params['initial'] = initial_func(i)
        states1 = sim_func(params)[100]

        params['N'] = 2*i
        params['dx'] = params['dx']/2
        #params['dt'] = params['dt']/2
        #params['nsteps'] = params['nsteps']*2
        params['initial'] = initial_func(2*i)
        states2 = sim_func(params)[100]

        params['dx'] = params['dx']*2
        #params['dt'] = params['dt']/2
        #params['nsteps'] = params['nsteps']*2

        states2_red = np.empty(i)
        #animate_states_1D(sim_func(params))
        for j in range(i):
            states2_red[j] = states2[2*j]
        rms_val.append(np.sqrt(np.sum((states1 - states2_red) ** 2) / i))
    sys.stdout.write('\n')
    plt.plot(np.linspace(params['dx']*lower, params['dx']*upper, upper-lower), np.array(rms_val))
    plt.xlabel("Grid Points")
    plt.ylabel("RMS Error")
    plt.savefig("burgers-convergence.png")
    plt.show()
    #return np.array(rms_val)

def spatial_order_1D(sim_func, initial_func, params):
    nodes = params['N']
    params['initial'] = initial_func(params['N'])
    states1 = sim_func(params)

    params['N'] = params['N']*2
    params['dx'] = params['dx']/2
    params['initial'] = initial_func(params['N'])
    states2 = sim_func(params)    
    
    params['N'] = params['N']*2
    params['dx'] = params['dx']/2
    params['initial'] = initial_func(params['N'])
    states4 = sim_func(params)
    
    order = np.empty(params['nsteps'])
    states2_red = np.empty(nodes)
    states4_red = np.empty(nodes)
    for i in range(params['nsteps']):
        for j in range(nodes):
            states2_red[j] = states2[i][2*j]
            states4_red[j] = states4[i][4*j]
        order[i] = np.sqrt(np.sum(np.divide(states1[i] - \
            states2_red, states2_red-states4_red)**2)/nodes)
    plt.ylabel(r"Spatial error $\propto\log[\varepsilon(\Delta x)]$")
    plt.ylim([-1, 4])
    plt.plot(np.arange(0, params['nsteps']), order)
    plt.savefig("space-order.png")
    plt.show()

def temporal_order_1D(sim_func, initial_func, params):
    nsteps = params['nsteps']
    params['initial'] = initial_func(params['N'])
    states1 = sim_func(params)   
    
    params['dt'] = params['dt']/2
    params['nsteps'] = params['nsteps']*2
    states2 = sim_func(params)
    
    params['dt'] = params['dt']/2
    params['nsteps'] = params['nsteps']*2
    states4 = sim_func(params)

    order = np.empty(nsteps)
    states2_red = np.empty(params['N'])
    states4_red = np.empty(params['N'])
    for i in range(nsteps):
       for j in range(params['N']):
           states2_red[j] = states2[2*i][j]
           states4_red[j] = states4[4*i][j]
       order[i] = np.sqrt(np.sum(np.divide(states1[i] - \
           states2_red, states2_red-states4_red)**2)/params['N'])
    plt.ylabel(r"Temporal order $\propto\log[\varepsilon(\Delta t)]$")
    plt.xlabel("Time Step")
    plt.ylim([-1, 4])
    plt.plot(np.arange(0, nsteps), order)
    plt.savefig("time-order.png")
    plt.show()

def variance_1D(states, dx, dt):    
    nsteps = len(states)
    nodes = states[0].shape[0]
    L = nodes*dx
    variance = []
    for i in range(nsteps):
        mean = np.sum(states[i])/nodes
        variance.append(np.sum((states[i] - mean)**2)/nodes)
    plt.ylabel("W")
    plt.xlabel("$t$")
    plt.xticks(np.linspace(0, dt*nsteps, 21))
    plt.plot(np.linspace(0, dt*nsteps, nsteps), np.sqrt(np.array(variance)))
    #plt.plot(np.arange(0, nsteps), 
    #    np.power(np.linspace(0, nsteps*dt, nsteps)*math.pow(nodes, -1.5), 1/3))
    plt.plot(np.linspace(0, dt*nsteps, nsteps), \
        .000125*math.pow(nodes, .5)*np.power(np.linspace(0, float(nsteps)*dt, nsteps)*math.pow(nodes, -1.5), 1./3))
    plt.plot(np.linspace(0, dt*nsteps, nsteps), \
        np.full(nsteps, .000013*math.pow(nodes, .5)))

    plt.legend(['Numerical Variance', r'Predicted Scaling Regime in 1$\times \sim 10^{-4}$', 
        r'Predicted Scaling Regime in 2 $\times \sim 10^{-5}$'])
    plt.savefig('kpz-scaling.png')
    #plt.show()

    


### Output Functions ###
def print_states(states):
    for s in states:
        print(s)

def animate_1D(states):
    fig, ax = plt.subplots()
    nodes = states[0].shape[0]
    x = np.linspace(0, +1.0, nodes)
    line, = ax.plot(x, states[0])
    ax.set_ylim([0, 2])
    
    def animate(i):
        line.set_ydata(states[i])
        return line,

    def init():
        line.set_ydata(np.ma.array(x, mask=True))
        return line,

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(states)),
        init_func=init, interval=25, blit=False)
    plt.ylabel("u (e.g. fluid flow)")
    plt.xlabel("x/L")
    plt.show()


def animate_2D(states):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, 1)

    nodes = states[0].shape[0]
    nsteps = len(states)

    l = np.linspace(-1.0, +1.0, nodes)
    (x, y) = np.meshgrid(l, l)
    
    current_plot = None
    counter = 0
    while True:
        # exit on window closing
        if not plt.fignum_exists(fig.number):
            break
        
        # remove previous plot
        if current_plot is not None:
            ax.collections.remove(current_plot)

        # increment and redraw
        counter = (counter + 1) % nsteps
        current_plot = ax.plot_surface(x, y, states[counter], rstride=int(max(1, nodes/20)), 
            cstride=int(max(1, nodes/20)), cmap=cm.coolwarm, antialiased=False)

        plt.pause(.001)
    

def plot_1D(state, name):
    nodes = state.shape[0]
    plt.plot(np.linspace(0, +1.0, nodes), state)
    plt.ylabel(r"$\varphi$")
    plt.xlabel("x/L")
    plt.ylim([0, 2])
    plt.savefig(name)
    plt.clf()

def plot_2D(state, name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, 1)

    nodes = state.shape[0]
    l = np.linspace(-1.0, +1.0, nodes)
    (x, y) = np.meshgrid(l, l)

    ax.plot_surface(x, y, state, rstride=int(max(1, nodes/20)),
        cstride=int(max(1, nodes/20)), cmap=cm.coolwarm, antialiased=False)
    ax.set_zlabel(r"$\varphi$")
    ax.set_xlabel("x/L")
    ax.set_ylabel("y/L")
    plt.savefig(name)
    plt.clf()



### main ###
def main(args):
    N = 100
    dx = .1
    params = {'N':N, 'dt':.001, 'dx':dx, 'nsteps':10000, 
        'initial':make_uniform_1D(N), 'nu':.1, 'rand':.01, 'lambda':1}
    #states = burgers_1D(params)
    #states = kpz_2D(params)
    #plot_2D(states[25])
    states = kpz_1D(params)
    #states = kpz_2D(params)
    #states = burgers_1D(params)
    #states = diffusion_1D(params)
    variance_1D(states, params['dx'], params['dt'])
    #animate_1D(states)
    #animate_2D(states)
    #plot_1D(states[0], "kpz3-0.png")
    #plot_1D(states[25], "kpz3-25.png")
    #plot_1D(states[50], "kpz3-50.png")
    #plot_1D(states[500], "kpz3-500.png")
    #plot_2D(states[0], "kpz2D-2-0.png")
    #plot_2D(states[5], "kpz2D-2-5.png")
    #plot_2D(states[10], "kpz2D-2-10.png")
    #plot_2D(states[15], "kpz2D-2-15.png")
    #plot_1D(states[500], "kpz2-500.png")
    #plot_1D(states[5000], "burgers2-5000.png")
    #print_states(states)
    #convergence_1D(burgers_1D, make_gaussian_1D, params)
    #spatial_order_1D(burgers_1D, make_gaussian_1D, params)
    #temporal_order_1D(burgers_1D, make_gaussian_1D, params)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
