#problem=VLMOP2
seed_num=3
#epo pmgda


for seed in $(seq 0 $(($seed_num-1)))
  do
    for problem in RE21 RE37
    do
#      for solver_name in agg_ls agg_tche agg_mtche agg_pbi agg_cosmos agg_softtche agg_softmtche
#      do
#        python run_syn_psl.py --solver-name $solver_name --draw-fig False --epoch 1000 --problem-name $problem --seed-idx $seed
#      done

      for solver_name in pmgda epo
      do
        python run_syn_psl.py --solver-name $solver_name --draw-fig False --epoch 100 --problem-name $problem --seed-idx 2
      done
  done
done

sleep 100