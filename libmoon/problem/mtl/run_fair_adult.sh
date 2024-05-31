totalepoch=20
dataset=adult
seednum=10
upp=50000



# Run set methods
for solver in hvgrad pmtl
  do
    for seed in $(seq 1 $seednum)
    do
      python run_fair_set.py --epoch $totalepoch --solver $solver --seed $seed --dataset $dataset
    done
done

for solver in uniform epo pmgda
  do
    for seed in $(seq 1 $seednum)
    do
      python run_fair_pref.py --epoch $totalepoch --solver $solver --seed $seed --dataset $dataset --uniform-update-iter $upp
    done
done

for agg in mtche ls tche pbi cosmos
#for agg in cosmos
  do
    for seed in $(seq 1 $seednum)
    do
      python run_fair_pref.py --epoch $totalepoch --solver agg --agg $agg --seed $seed --dataset $dataset --uniform-update-iter $upp
    done
done

#





sleep 100



