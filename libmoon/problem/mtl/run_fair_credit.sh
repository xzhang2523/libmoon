#totalepoch=1200
totalepoch=20
upp=3000

dataset=credit
seednum=1

## Run set methods
for agg in tche ls pbi
#for agg in cosmos
  do
    for seed in $(seq 1 $seednum)
    do
      python run_fair_pref.py --epoch $totalepoch --solver agg --agg $agg --seed $seed --dataset $dataset --uniform-update-iter $upp
    done
done


for solver in uniform epo pmgda
  do
    for seed in $(seq 1 $seednum)
    do
      python run_fair_pref.py --epoch $totalepoch --solver $solver --seed $seed --dataset $dataset --uniform-update-iter $upp
    done
done

#


for solver in hvgrad
  do
    for seed in $(seq 1 $seednum)
    do
      python run_fair_set.py --epoch $totalepoch --solver $solver --seed $seed --dataset $dataset
    done
done


sleep 100



