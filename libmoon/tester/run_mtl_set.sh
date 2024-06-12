
epoch=100
nprob=5

for dataset in adult
  do
  for solver in  moosvgd pmtl hvgrad
  do
    for architecture in M1
    do
      python run_mtl_clean_set.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob --solver $solver
    done
  done

done




sleep 100









