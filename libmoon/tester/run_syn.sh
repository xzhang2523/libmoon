
niter=100

for solver in pmgda epo
do
  for seed in 0
  do
    python run_syn_clean.py --solver $solver --seed $seed --problem-name ZDT1 --n-iter $niter
  done
done

for agg in mtche cosmos
do
  for seed in 0
  do
    python run_syn_clean.py --solver agg --seed $seed --problem-name ZDT1 --agg $agg --n-iter $niter
  done
done


sleep 100