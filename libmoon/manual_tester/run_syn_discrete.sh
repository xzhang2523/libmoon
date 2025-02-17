
problem=regression
seed_num=3

# mgdaub random epo pmgda
# preference-based: agg_ls agg_tche agg_pbi agg_cosmos, agg_softtche
# set-based pmtl hvgrad moosvgd
# zxy method: pmgda, uniform.
for seed in $(seq 0 $(($seed_num)))
do
#  for solver_name in mgdaub random epo pmgda agg_ls agg_tche agg_mtche agg_pbi agg_cosmos agg_softtche pmtl hvgrad moosvgd
  for solver_name in mgdaub random epo agg_ls agg_tche agg_mtche agg_pbi agg_cosmos agg_softtche pmtl hvgrad moosvgd
  do
    python run_syn_discrete.py --solver-name $solver_name --draw-fig False --epoch 1000 --problem-name $problem --seed-idx $seed
  done
done



sleep 100