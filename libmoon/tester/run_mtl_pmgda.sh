epoch=100
nprob=10
PaperName=TETCI


for dataset in credit
  do
  for solver in pmgda
  do
    for architecture in M1
    do
      python run_mtl_clean_pref.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob \
       --solver $solver --PaperName $PaperName
    done
  done
done


sleep 100









