
problem=RE37

for solver_name in agg_ls agg_cosmos agg_mtche epo
do
  python run_syn_psl.py --solver-name $solver_name --draw-fig False --epoch 200 --problem-name $problem
done


sleep 100