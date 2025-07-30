import pickle as pkl
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram


if __name__ == '__main__':
    # load data: 
    with open('nonintegrable/seed_42.pkl', 'rb') as f:
        data = pkl.load(f)

    # the structure of the data looks like this:
    # [{'00000': xxx, '00001': xxx, '00010': xxx, '00011': xxx ...}, ...]

    # total 5000 circuits, each with 1024 shots
    # first 1296 circuits       -->     SPAM
    # the rest 3704 circuits    -->     TOMO
    # the order is the same as the given circuit
    
    plot_histogram(data[1296], title='TOMO circuit 0')
    plt.tight_layout()
    plt.show()


