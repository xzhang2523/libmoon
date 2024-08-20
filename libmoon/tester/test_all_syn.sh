for solver in mgdaub epo random
do
  python run_syn_all.py --solver $solver --seed 1
done


sleep 100