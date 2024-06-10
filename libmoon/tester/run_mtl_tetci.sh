epoch=100
nprob=10
PaperName=TETCI


for dataset in adult compass credit
  do
    for seed in 4 5
    do
      for agg in mtche cosmos
      do
        for architecture in M2
        do
          python run_mtl_clean_pref.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob \
           --solver agg --agg $agg --PaperName $PaperName --seed $seed --sigma 0.5
        done
      done

      for solver in pmgda epo
      do
        for architecture in M2
        do
          python run_mtl_clean_pref.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob \
           --solver $solver --PaperName $PaperName --seed $seed --sigma 0.5
        done
      done
    done
done




sleep 100









