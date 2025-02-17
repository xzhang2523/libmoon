for solver in PMGDA EPO MOOSVGD GradHV PMTL GradAgg MGDAUB
do
  python libmoon/auto_tester/test_synthetic.py --solver-name $solver
done
sleep 100