epoch=60
nprob=10
PaperName=TETCI
sigma=0.4


for dataset in credit
  do
    for seed in 1
    do
      for agg in cosmos mtche
      do
        for architecture in M2
        do
          python run_mtl_clean_pref.py --architecture $architecture --dataset $dataset --epoch $epoch --n-prob $nprob \
           --solver agg --agg $agg --PaperName $PaperName --seed $seed --sigma $sigma --cosmos-hp 40
        done
      done

      for solver in pmgda epo
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









