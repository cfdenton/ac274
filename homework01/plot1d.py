import sys
import numpy as np
from vispy import gloo, app
import argparse

def main(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()
    filename = args.filename
    values = load_data(filename) 
    c = Canvas(values, filename)
    app.run()
   

def load_data(filename):
    values = []
    with open(filename, 'r') as f:
        for line in f:
            values.append(float(line))    
    return np.array(values, dtype=np.float32)



vertex_shader = '''
attribute vec2 position;
uniform float scale;
void main(void) {
    vec2 scaled_position = position;
    scaled_position.y = position.y*scale; 
    gl_Position = vec4(scaled_position, 0.0, 1.0);
}
'''

fragment_shader = '''
void main(void) {
    gl_FragColor = vec4(0.1, 0.1, 0.1, .3);
}
'''

class Canvas(app.Canvas):
    def __init__(self, val, filename):
        app.Canvas.__init__(self, size=(800, 800), keys='interactive', title=filename)
        self.program = gloo.Program(vertex_shader, fragment_shader)
        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        N = val.size
        x = np.linspace(-1.0, +1.0, N).astype(np.float32)
        self.pos = np.c_[x, val].astype(np.float32)
        self.maximum = np.amax(np.absolute(val))
        
        #print(self.pos)
        self.show()

    def on_resize(self, event):
        width, height = event.physical_size
        gloo.set_viewport(0, 0, width, height)

    
    def on_draw(self, event):
        gloo.clear('white')
        gloo.set_viewport(0, 0, *self.physical_size)

        self.program['position'] = self.pos
        self.program['scale'] = .75/self.maximum 
        print("scale: " + str(.75/self.maximum))
        self.program.draw('line_strip')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
