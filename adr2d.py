import sys
import numpy as np
from vispy import gloo, app
import time


# sample texture
radius = 32
im1 = np.random.normal(
    0.8, 0.3, (radius * 2 + 1, radius * 2 + 1)).astype(np.float32)

# mask to disk
L = np.linspace(-radius, radius, 2*radius + 1)
(X, Y) = np.meshgrid(L, L)
im1 *= np.array((X ** 2 + Y ** 2) <= radius ** 2, dypte='float32')

# number of nodes per dimension (N^2 total)
N = 5 

vertex_shader = """
attribute vec2 position;
void main() {
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

fragment_shader = """
void main() {
    gl_FragColor = (1.0, 0, 0, 1.0);
}
"""

class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='Advection-Diffusion-Reaction 2D',
                            size=(512, 512), keys='interactive')
        
        x = np.linspace(-1.0, +1.0, N)
        y = np.linspace(-1.0, +1.0, N)
        xc, yc = np.meshgrid(x, y) 
        self._program = gloo.Program(vertex_shader, fragment_shader)
        vert = np.c_[np.matrix.flatten(xc), np.matrix.flatten(yc)].astype(np.float32)
        print(vert)
        self._program["position"] = vert
        self.show()

    
    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

        
    def on_draw(self, event):
        gloo.clear((1, 1, 1, 1))
        gloo.set_viewport(0, 0, *self.physical_size)
        self._program.draw('triangle_fan')
        # draw canvas


def main(arguments):
    c = Canvas()
    app.run()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
