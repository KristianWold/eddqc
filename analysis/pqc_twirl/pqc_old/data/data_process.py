import pickle as pkl
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram


if __name__ == '__main__':
    # load data: 
    with open('L=8.pkl', 'rb') as f:
        data = pkl.load(f)

    # the structure of the data looks like this:
    # [{'0000': xxx, '0001': xxx, '0010': xxx, '0011': xxx ...}, ...]

    # total 5000 circuits, each with 1024 shots
    # first 1296 circuits       -->     SPAM
    # the rest 3704 circuits    -->     TOMO
    # the order is the same as the given circuit
    
    plot_histogram(data[1296], title='TOMO circuit 0')
    plt.tight_layout()
    plt.show()


