


for seed in {1..3}
do
  for solver in pmgda epo
  do
    python run_syn_clean.py --seed-idx $seed --solver $solver
  done
done