epoch=20
nprob=10
PaperName=TETCI
sigma=0.4


for dataset in adult credit compass
  do
    for seed in 1 2 3 4 5
    do
      for solver in pmgda
      do
        for architecture in M2
        do
          python run_mtl_clean_pref.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob \
           --solver $solver --PaperName $PaperName --seed $seed --sigma $sigma
        done
      done
    done
done




sleep 100









