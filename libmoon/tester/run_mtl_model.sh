epoch=100
nprob=10
PaperName=TETCI
sigma=0.4


for dataset in adult compass
  do
    for solver in pmgda
    do
      for architecture in M1 M2 M3 M4
      do
        python run_mtl_clean_pref.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob \
         --solver $solver --PaperName $PaperName --seed 1 --sigma $sigma
      done
    done
done




sleep 100









