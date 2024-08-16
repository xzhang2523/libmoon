import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    bars = plt.bar(['CPU', 'RTX4060', 'RTX3080','RTX4090'],
            np.round(np.array([2542.5, 740.6, 472.73, 196.17])/60,2), width=0.7)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel('Device', fontsize=20)
    plt.ylabel('Time (min)', fontsize=20)
    plt.bar_label(bars, fontsize=18)
    # plt.title('Time to train 10 epochs', fontsize=20)
    # plt.legend()
    plt.savefig('time.pdf', bbox_inches='tight')
    plt.savefig('time.svg', bbox_inches='tight')
    plt.show()
