
problem=VLMOP1

# mgdaub random epo

for seed in {0..1}
do
  for solver_name in agg_ls agg_tche agg_pbi agg_cosmos
  do
    python run_syn_discrete.py --solver-name $solver_name --draw-fig False --epoch 1000 --problem-name $problem --seed-idx $seed
  done
done
sleep 100