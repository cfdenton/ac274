import sys
import numpy as np
from vispy import gloo, app
import time
import argparse


# number of nodes per dimension (N^2 total)

vertex_shader = """
attribute vec2 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_texcoord = texcoord; 
}
"""

fragment_shader = """
uniform sampler2D texture;
varying vec2 v_texcoord;
void main() {
    float v;
    v = texture2D(texture, v_texcoord).r;
    gl_FragColor = vec4(1.0-v, 1.0-v, 1.0-v, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self, state, transfer):
        app.Canvas.__init__(self, title='Advection-Diffusion-Reaction 2D',
                            size=(512, 512), keys='interactive')
        
        # make grid
        self._program = gloo.Program(vertex_shader, fragment_shader)
        self._timer = app.Timer('auto', connect=self.update, start=True)
        
        self.state = state
        self.transport = transfer
        #print(state.size)

        self._program["position"] = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
        self._program["texcoord"] = [(0, 0), (0, 1), (1, 0), (1, 1)]
        self._program["texture"] = self.vec_to_tex(self.state)
        self.show()

    
    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

        
    def on_draw(self, event):
        # draw canvas
        gloo.clear('white')
        gloo.set_viewport(0, 0, *self.physical_size)
    
        #print(self.state)

        self.state = next_state(self.state, self.transport)
        self._program["texture"] = self.vec_to_tex(self.state) 

        #print(self.state)
        self._program["texture"].interpolation = 'linear'

        self._program.draw('triangle_strip')


    def vec_to_tex(self, state):
        tex = np.resize(state, (N, N, 1))
        tex = np.repeat(tex, 4, 2)
        tex[:, :, 3].fill(1)
        return tex


def next_state(state, transport):
    return np.matmul(transport, state)
    

# initial circle
def make_initial():
    L = np.linspace(-1.0, 1.0, N)
    (x, y) = np.meshgrid(L, L)
    state = .5 * np.array((x ** 2 + y ** 2) <= radius * radius, dtype=np.float32)
    state = state.flatten()
    return state

def make_transport():
    transport = np.zeros((N*N, N*N), dtype=np.float32) 
    for i in range(N*N):
        transport[i][i] = 1 - 4*D*dt/(d*d) + R*dt
        # right 
        transport[i][(i-1)%(N*N)] = D*dt/(d*d) - Ux*dt/(2*d)
        # left 
        transport[i][(i+1)%(N*N)] = D*dt/(d*d) + Ux*dt/(2*d)
        # up
        transport[i][(i+N)%(N*N)] = D*dt/(d*d) - Uy*dt/(2*d)
        # down
        transport[i][(i-N)%(N*N)] = D*dt/(d*d) + Uy*dt/(2*d)
    return transport
      


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
    args = parser.parse_args()
    global N, D, Ux, Uy, R, radius, dt, d
    N = args.N
    D = args.D
    Ux = args.Ux
    Uy = args.Uy
    R = args.R
    radius = args.radius
    dt = args.dt
    d = args.d

    delta = D*dt/(d*d)
    alphaX = Ux*dt/d
    alphaY = Uy*dt/d
    rho = R*dt

    state = make_initial()
    transfer = make_transport()
     
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
    

    c = Canvas(state, transfer)
    app.run()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
