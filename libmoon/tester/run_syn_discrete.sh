
problem=VLMOP1

for seed in {1..2}
do
  for solver_name in mgdaub random epo
  do
    python run_syn_discrete.py --solver-name $solver_name --draw-fig False --epoch 200 --problem-name $problem --seed-idx $seed
  done
done
sleep 100