
epoch=100

for seed in {1..5}
do
  for dataset in mnist fashion fmnist
  do

    for solver in pmgda epo
    do

        python run_mtl_clean.py --dataset $dataset --epoch $epoch --n-prob 5 --solver $solver --agg mtche --seed $seed --batch-size 2048
    done

    for agg in cosmos mtche
    do
        python run_mtl_clean.py --dataset $dataset --epoch $epoch --n-prob 5 --solver agg --agg $agg --seed $seed --batch-size 2048
    done

  done
done

sleep 1000