for solver in pmgda
do
  for seed in 0 1 2 3 4
  do
    python run_syn_clean.py --solver $solver --seed $seed
  done
done
