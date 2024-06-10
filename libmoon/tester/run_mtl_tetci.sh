epoch=100
nprob=10
PaperName=TETCI


for dataset in adult compass credit
  do
  for agg in mtche cosmos
  do
    for architecture in M2
    do
      python run_mtl_clean_pref.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob \
       --solver agg --agg $agg --PaperName $PaperName
    done
  done

  for solver in pmgda epo
  do
    for architecture in M2
    do
      python run_mtl_clean_pref.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob \
       --solver $solver --PaperName $PaperName
    done
  done

done




sleep 100









