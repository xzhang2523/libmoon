for dataset in adult
    do
    for solver_name in mgdaub random epo pmgda agg_ls agg_tche agg_pbi agg_cosmos agg_softtche pmtl hvgrad moosvgd
    do
      python run_mtl_discrete.py --problem-name adult --solver-name $solver_name --use-plt False
    done
done

sleep 100