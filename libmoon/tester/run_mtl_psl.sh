seed_num=3


for seed in $(seq 0 $(($seed_num)))
do
  for dataset in mnist
      do
      for solver in agg_ls agg_tche agg_mtche agg_pbi agg_cosmos agg_softtche agg_softmtche
      do
        python run_mtl_psl.py --solver-name $solver --problem-name $dataset --seed-idx $seed
      done
  done
done

sleep 100