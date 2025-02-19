for solver in PMGDA EPO MOOSVGD GradHV PMTL GradAgg MGDAUB
#do
#  python libmoon/auto_tester/test_synthetic.py --solver-name $solver --n-epoch 2000
#done
do
  python test_synthetic.py --solver-name $solver --n-epoch 10000
done

for solver in PMGDA EPO MOOSVGD GradHV PMTL GradAgg MGDAUB
do
  python libmoon/auto_tester/test_mtl.py --solver-name $solver
done

sleep 100