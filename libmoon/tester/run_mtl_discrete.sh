seed_num=3


for seed in $(seq 1 $(($seed_num - 1)))
  do
    for solver_name in mgdaub random epo pmgda agg_ls agg_tche agg_mtche agg_pbi agg_cosmos agg_softtche pmtl hvgrad moosvgd
#    for solver_name in mgdaub
    do
      python run_mtl_discrete.py --problem-name adult --solver-name $solver_name --use-plt False --epoch 10 --seed-idx $seed
    done
done

sleep 100