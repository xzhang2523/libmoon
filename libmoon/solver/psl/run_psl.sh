for dataset in mnist fmnist fashion
  do
    for agg in cosmos mtche pbi ls
    do
      python run_mtl_psl.py --dataset $dataset --agg $agg
    done
done