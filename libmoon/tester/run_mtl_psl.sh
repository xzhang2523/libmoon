seed_num=2


for seed in $(seq 1 $(($seed_num)))
do
  for dataset in mnist
      do
      for solver in agg_ls agg_tche agg_pbi agg_cosmos agg_softtche
      do
        python run_mtl_psl.py --solver-name $solver --problem-name $dataset --seed-idx $seed --epoch 40
      done
  done
done

sleep 100