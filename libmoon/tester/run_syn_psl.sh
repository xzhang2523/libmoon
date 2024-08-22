problem=RE21
seed_num=3

#epo pmgda
for seed in $(seq 2 $(($seed_num)))
  do
#  for solver_name in agg_ls agg_tche agg_mtche agg_pbi agg_cosmos agg_softtche
  for solver_name in epo pmgda
  do
    python run_syn_psl.py --solver-name $solver_name --draw-fig False --epoch 100 --problem-name $problem --seed-idx $seed
  done
done

sleep 100