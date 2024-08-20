
problem=VLMOP1

for solver_name in agg_soft_tche
do
  python run_syn_psl.py --solver-name $solver_name --draw-fig False --epoch 200 --problem-name $problem
done


sleep 100