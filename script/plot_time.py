

from matplotlib import pyplot as plt


if __name__ == '__main__':
    plt.bar(['CPU', 'RTX2080', 'RTX4060', 'RTX4090'], [2542.5, 0, 750.4, 4], width=0.7)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel('Device', fontsize=20)
    plt.ylabel('Time (s)', fontsize=20)
    # plt.title('Time to train 10 epochs', fontsize=20)

    plt.savefig('time.pdf', bbox_inches='tight')
    plt.show()
