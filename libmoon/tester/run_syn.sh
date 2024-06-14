
niter=2000


for seed in 0 1 2 3 4
  do
  for problem_name in VLMOP1 VLMOP2
  do
      for solver in pmgda epo
      do
        python run_syn_clean.py --solver $solver --seed $seed --problem-name $problem_name --n-iter $niter
      done

      for agg in mtche cosmos
      do
        python run_syn_clean.py --solver agg --seed $seed --problem-name $problem_name --agg $agg --n-iter $niter
      done
    done
done

sleep 100