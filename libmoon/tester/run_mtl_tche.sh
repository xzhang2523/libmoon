
for agg in cosmos mtche
do
  for dataset in adult
  do
    for architecture in M1
    do
      python run_mtl_clean.py --architecture $architecture --dataset $dataset --epoch 20 --n-prob 10 --solver agg --agg $agg
    done
  done
done

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
epoch=800
nprob=5

for dataset in adult compass credit
  do
  for agg in cosmos mtche pbi
  do
    for architecture in M1
    do
      python run_mtl_clean.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob --solver agg --agg $agg
    done
  done

  for solver in pmgda epo
  do
    for architecture in M1
    do
      python run_mtl_clean.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob --solver $solver
    done
  done

done




sleep 100









