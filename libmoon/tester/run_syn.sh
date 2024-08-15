
niter=2000


for seed in 0
  do
  for problem_name in VLMOP1 VLMOP2
  do
      for solver in epo
      do
        python run_syn_clean.py --solver $solver --seed $seed --problem-name $problem_name --n-iter $niter
      done
  done
done




sleep 100