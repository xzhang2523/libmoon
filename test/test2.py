import numpy as np



if __name__ == '__main__':
    categories = ['apple', 'banana', 'kiwi']
    probabilities = [0.2, 0.2, 0.6]

    # draw 1000 samples
    n = 1000
    draw = np.random.choice(categories, n, p=probabilities)

    # print counts to verify
    from collections import Counter
    print(Counter(draw))