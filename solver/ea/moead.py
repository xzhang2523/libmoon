import numpy as np
import argparse

'''
    MOEA/D framework, use different evolutionary operators.
'''

class MOEAD_base:
    def __init__(self):
        pass

    def update(self):
        print()




class MOEAD(MOEAD_base):
    '''
        MOEAD.
    '''
    def __init__(self, n_gen=200):
        self.n_gen = n_gen

    def solve(self, problem):
        for idx in range(self.n_gen):
            print()



    def update(self):
        pass




if __name__ == '__main__':
    from problem.synthetic.zdt import ZDT1
    problem = ZDT1()
    solver = MOEAD()

    solver.solve(problem)

    print()



