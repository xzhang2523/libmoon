import pickle

def save_pickle(res, filename):
    with open(filename, 'wb') as f:
        pickle.dump(res, f)