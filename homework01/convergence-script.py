import numpy as np
import sys

def main(arguments):
    with open('convergence.dat', 'w') as converge:
        for i in range(10, 251, 10):
            data1 = []
            data2 = []
            with open('adr1d-500-grid-'+str(i)+'.dat', 'r') as f:
                for line in f:
                    data1.append(float(line))
            with open('adr1d-500-grid-'+str(2*i)+'.dat', 'r') as f:
                for line in f:
                    data2.append(float(line))
            arr1 = np.array(data1)   
            arr2 = np.array([data2[i] for i in range(0, len(data2), 2)])
            diff = np.linalg.norm(arr1 - arr2)
            converge.write(str(i) + ' ' + str(diff) + '\n')
            

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
