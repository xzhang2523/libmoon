
#for agg in cosmos mtche
#do
#  for dataset in adult
#  do
#    for architecture in M1 M2 M3 M4
#    do
#      python run_mtl_clean.py --architecture $architecture --dataset $dataset --epoch 20 --n-prob 10 --solver agg --agg $agg
#    done
#  done
#done




epoch=20

#for agg in mtche
#do
#  for dataset in credit
#  do
#    for architecture in M1
#    do
#      python run_mtl_clean.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob 10 --solver agg --agg $agg
#    done
#  done
#done


for solver in pmgda
do
  for dataset in credit
  do
    for architecture in M1
    do
      python run_mtl_clean.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob 10 --solver $solver
    done
  done
done


sleep 100









