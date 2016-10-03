import sys
import numpy as np
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
def make_gaussian(nodes):
    return np.exp(-(np.linspace(-1.0, +1.0, nodes)**2)*10)

def make_uniform(nodes):
    return np.full(nodes, .5)

def make_gaussian_2D(nodes):
    l = np.linspace(-1.0, +1.0, nodes)
    (x, y) = np.meshgrid(l, l)
    return np.exp(-x**2 -y**2)


### Check Functions ###
def convergence_1D(sim_func, initial_func):
    params = {'dt':.01, 'nsteps':101, 'nu':.1}
    dx_base = .1
    rms_val = []
    lower, upper = 10, 1000
    for i in range(lower, upper):
        sys.stdout.write('\rChecking convergence for N = ' + str(i))
        params['N'] = i
        params['dx'] = dx_base
        params['initial'] = initial_func(i)
        states1 = sim_func(params)[100]
        params['N'] = 2*i
        params['dx'] = dx_base/2
        params['initial'] = initial_func(2*i)
        states2 = sim_func(params)[100]
        states2_red = np.empty(i)
        #animate_states_1D(sim_func(params))
        for j in range(i):
            states2_red[j] = states2[2*j]
        rms_val.append(np.sqrt(np.sum((states1 - states2_red) ** 2) / i))
    sys.stdout.write('\n')
    plt.plot(np.arange(lower, upper), np.array(rms_val)) 
    plt.show()
    #return np.array(rms_val)


### Output Functions ###
def print_states(states):
    for s in states:
        print(s)

def animate_1D(states):
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

    ani = animation.FuncAnimation(fig, animate, np.arange(1, len(states)),
        init_func=init, interval=25, blit=False)
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
    

def plot_1D(state):
    nodes = state.shape[0]
    plt.plot(np.linspace(-1.0, +1.0, nodes), state)
    plt.show()

def plot_2D(state):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    nodes = state.shape[0]
    l = np.linspace(-1.0, +1.0, nodes)
    (x, y) = np.meshgrid(l, l)

    ax.plot_surface(x, y, state, rstride=int(max(1, nodes/20)),
        cstride=int(max(1, nodes/20)), cmap=cm.coolwarm, antialiased=False)
    plt.show()



### main ###
def main(args):
    N = 40
    dx = 1
    params = {'N':N, 'dt':.01, 'dx':dx, 'nsteps': 50, 
        'initial':make_gaussian_2D(N), 'nu':1, 'rand':.4, 'lambda':1}
    states = kpz_2D(params)
    plot_2D(states[25])
    #states = kpz_1D(params)
    #animate_1D(states)
    #plot_1D(states[0])
    #print_states(states)
    #convergence_1D(burgers_1D, make_gaussian)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
