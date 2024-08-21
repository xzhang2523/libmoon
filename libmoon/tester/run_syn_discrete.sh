
problem=VLMOP1
seed_num=2

# mgdaub random epo
# preference-based: agg_ls agg_tche agg_pbi agg_cosmos, agg_softtche
# set-based pmtl hvgrad moosvgd
# zxy method: pmgda, uniform.


for seed in $(seq 0 $(($seed_num - 1)))
do
  for solver_name in agg_softtche
  do
    python run_syn_discrete.py --solver-name $solver_name --draw-fig False --epoch 1000 --problem-name $problem --seed-idx $seed
  done
done
sleep 100