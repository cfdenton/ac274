import sys
import numpy as np
from vispy import gloo, app
import time

N = 1000 # number of nodes 
U = 0.1 
D = 2.5e-4 
R = 0
dx = 1e-3
dt = 1e-3
delta = D*dt/(dx * dx)
alpha = U*dt/dx
rho = R*dt

mass = []
average_pos = []
variance = []
entropy = []

def main(arguments):
    print("delta: " + str(delta))
    print("alpha: " + str(alpha))
    print("rho: " + str(rho))
    print("delta = Dh/d^2 < 1/2 for stability")
    print("alpha = Uh/d < 2delta < 1 for stability")
    try: 
        c = Canvas()
        app.run()
    finally:
        output_val(mass, 'mass1d.csv')
        output_val(average_pos, 'avepos1d.csv')
        output_val(variance, 'variance1d.csv')
        output_val(entropy, 'entropy1d.csv')
    

def make_transport():
    transport = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        transport[i][(i-1) % N] = dt*(D/(dx*dx) - U/(2*dx))
        transport[i][i] = 1 - 2*D*dt/(dx * dx) + R*dt
        transport[i][(i+1) % N] = dt*(D/(dx*dx) + U/(2*dx))
    return transport

def compute_mass(state):
    return np.sum(state)

def compute_ave_pos(state):
    return np.sum(state*np.linspace(-1.0, +1.0, state.size))

def compute_var(state):
    x = np.linspace(-1.0, +1.0, state.size)
    ave = np.sum(state*x)
    return np.sum(state*x*x) - ave*ave

def compute_entropy(state):
    return -np.sum(state*state)

def update_diag(state):
    mass.append(compute_mass(state))
    average_pos.append(compute_ave_pos(state))
    variance.append(compute_var(state))
    entropy.append(compute_entropy(state))

def output_val(values, filename):
    with open(filename, 'w') as f:
        for i in values:
            f.write(str(i) + '\n')


VERT_SHADER = """
attribute vec2 a_position; 
void main(void) {
    gl_Position = vec4(a_position, 0.0, 0.3); 
}
"""

FRAG_SHADER = """
void main(void) {
    gl_FragColor = vec4(0.5, 0.5, 0.5, 1.0);
}
"""


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, keys='interactive', size=(800, 600))

        self._program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self._starttime = time.time()

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        #gloo.set_state(blend=True, clear_color='black',
        #               blend_func=('src_alpha', 'one'))
        self._timer = app.Timer('auto', connect=self.update, start=True)
        
        # build initial state
        self.state = np.zeros(N, dtype=np.float32)
        self.state[int(N/2)] = .5
        self.transport = make_transport()
        print(self.transport)
        self.show()


    def on_resize(self, event): 
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)



    def on_draw(self, event):
        gloo.clear((1, 1, 1, 1))
        if time.time() - self._starttime > dt:
            self._next_state() 
            self._starttime = time.time() 
        #print(np.c_[np.linspace(-1.0, +1.0, N), self.state])
        self._program['a_position'] = np.c_[
            np.linspace(-1.0, +1.0, N), self.state].astype(np.float32)
        self._program.draw('line_strip')


    def _next_state(self):      
        self.state = np.matmul(self.transport, self.state)
        update_diag(self.state)
    

if __name__ == '__main__':
   sys.exit(main(sys.argv[1:]))
