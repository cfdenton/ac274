import sys
import numpy as np
from vispy import gloo, app
import time
import argparse


def main(arguments):
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('-N', type=int, default='100')
    parser.add_argument('-D', type=float, default='.1')
    parser.add_argument('-Ux', type=float, default='0')
    parser.add_argument('-Uy', type=float, default='0')
    parser.add_argument('-R', type=float, default='0')
    parser.add_argument('-radius', type=float, default='.2')
    parser.add_argument('-dt', type=float, default='1')
    parser.add_argument('-d', type=float, default='1')
    parser.add_argument('-nsteps', type=int, default='1000')
    parser.add_argument('-chem', default='linear', help='linear, logistic')
    parser.add_argument('-a', type=float, default='0')
    parser.add_argument('-b', type=float, default='0')
    parser.add_argument('-initial', default='gaussian', help='gaussian, random')
    args = parser.parse_args()
    global N, D, Ux, Uy, R, radius, dt, d, nsteps, chem, a, b, initial
    N = args.N
    D = args.D
    Ux = args.Ux
    Uy = args.Uy
    R = args.R
    radius = args.radius
    dt = args.dt
    d = args.d
    nsteps = args.nsteps
    chem = args.chem
    a, b = args.a, args.b
    initial = args.initial
    if chem == 'rand-nl':
        a = np.random.uniform(-a, a, (N*N)).astype(np.float32)

    delta = D*dt/(d*d)
    alphaX = Ux*dt/d
    alphaY = Uy*dt/d
    rho = R*dt

    sys.stdout.write("Making transport matrix... ")
    transport = make_transport()
    sys.stdout.write("done.\n")

    states = []

    sys.stdout.write("Making initial configuration... ")
    states.append(make_initial())
    sys.stdout.write("done.\n")

    for i in range(nsteps):
        sys.stdout.write("\rCalculating states... current: " + str(i+1))
        states.append(next_state(states[i], transport))
    sys.stdout.write("\rCalculating states... done.\n")

    sys.stdout.write("Building visualization.\n")
    c = Canvas(states)
    app.run()

    print("delta: " + str(delta))
    print("alpha_x: " + str(alphaX))
    print("alpha_y: " + str(alphaY))
    print("rho: " + str(rho))
    print("delta = Dh/d2 < 1/2 for stability")
    print("alpha_j = U_j h/d < 2delta < 1 for stability")
    print("rho = Rh < 2delta")
    print("") 
    print("parameters:")
    print("N: " + str(N))
    print("D: " + str(D))
    print("Ux: " + str(Ux))
    print("Uy: " + str(Uy))
    print("R: " + str(R))
    print("radius: " + str(radius))
    print("h: " + str(dt))
    print("d: " + str(d))
    print("a: " + str(a))
    print("b: " + str(b))


# initial circle
def make_initial():
    L = np.linspace(-1.0, 1.0, N)
    (x, y) = np.meshgrid(L, L)
    if initial == 'gaussian':
        state = np.exp(np.array(-x ** 2 - y**2)/(radius * radius)).astype(np.float32).flatten()
    if initial == 'random':
        state = np.random.uniform(0.0, 0.01, (N*N)).astype(np.float32) 
    if initial == 'circle':
        state = np.array((x ** 2 + y ** 2) <= radius * radius, dtype=np.float32).flatten()
        #state=state * np.array((x ** 2 + y ** 2) <= radius * radius, dtype=np.float32).flatten()
    #state = state.flatten()
    return state

def make_transport():
    transport = np.zeros((N*N, N*N), dtype=np.float32) 
    for i in range(N*N):
        transport[i][i] = 1 - 4*D*dt/(d*d)
        # right 
        transport[i][(i-1)%(N*N)] = D*dt/(d*d) - Ux*dt/(2*d)
        # left 
        transport[i][(i+1)%(N*N)] = D*dt/(d*d) + Ux*dt/(2*d)
        # up
        transport[i][(i+N)%(N*N)] = D*dt/(d*d) - Uy*dt/(2*d)
        # down
        transport[i][(i-N)%(N*N)] = D*dt/(d*d) + Uy*dt/(2*d)
    return transport
 

def next_state(state, transport):
    if chem == 'linear':
        return np.matmul(transport, state) + R*state
    elif chem == 'logistic':
        return np.matmul(transport, state) + R*state*(1-state)
    elif chem == 'rand-nl':
        return np.matmul(transport, state) + a*state - b*state*state
    
     

    
vertex_shader = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
uniform float done;
varying float v_done;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord; 
    v_done = done;
}
"""

fragment_shader = """
uniform sampler2D texture;
varying vec2 v_texcoord;
varying float v_done;
void main() {
    float v;
    if (v_done == 0) {
        v = texture2D(texture, v_texcoord).r;
        gl_FragColor = vec4(1.0, 1.0 - v, 0.0, 1.0);
    }
    else if (v_done == 1) {
        gl_FragColor = vec4(1.0, 1.0, 1.0, 1.0); 
    }
}
"""

class Canvas(app.Canvas):
    def __init__(self, states):
        app.Canvas.__init__(self, title='Advection-Diffusion-Reaction 2D',
                            size=(512, 512), keys='interactive')
        
        # make grid
        self._program = gloo.Program(vertex_shader, fragment_shader)
        self._timer = app.Timer('auto', connect=self.update, start=True)
        
        self.states = states
        self.nstate = 0

        self._program["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self._program["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self._program["texture"] = self.vec_to_tex(self.states[self.nstate])
        self.show()

    
    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

        
    def on_draw(self, event):
        # draw canvas
        gloo.clear('white')
        gloo.set_viewport(0, 0, *self.physical_size)
    
        #print(self.state)
        if self.nstate < nsteps:
            self.nstate += 1
            self._program["texture"] = self.vec_to_tex(self.states[self.nstate]) 
            self._program["done"] = 0
        else:
            self._program["texture"] = self.vec_to_tex(np.zeros(N*N).astype(np.float32))
            self._program["done"] = 1

        #print(self.state)
        #self._program["texture"].interpolation = 'linear'

        self._program.draw('triangle_strip')


    def vec_to_tex(self, state):
        tex = np.resize(state, (N, N, 1))
        tex = np.repeat(tex, 4, 2)
        tex[:, :, 3].fill(1)
        return tex



if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
