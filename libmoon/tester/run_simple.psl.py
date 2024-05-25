# xs.shape: (batch_size, n_var)



from libmoon.solver.psl.core_psl import AggPSL



if __name__ == '__main__':

    agg_psler = AggPSL(problem, batch_size, device, epoch=1000, use_es=False)

    print()




