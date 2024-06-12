# for agg in cosmos mtche
#do
#  for dataset in adult
#  do
#    for architecture in M1 M2 M3 M4
#    do
#      python run_mtl_clean.py --architecture $architecture --dataset $dataset --epoch 20 --n-prob 10 --solver agg --agg $agg
#    done
#  done
#done

# for agg in mtche
#do
#  for dataset in credit
#  do
#    for architecture in M1
#    do
#      python run_mtl_clean.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob 10 --solver agg --agg $agg
#    done
#  done
#done

epoch=100

for seed in {1..1}
do
  for dataset in mnist
  do
    for agg in tche
    do
        python run_mtl_clean.py --dataset $dataset --epoch $epoch --n-prob 5 --solver agg --agg $agg --seed $seed --batch-size 2048
    done

  done
done

sleep 1000