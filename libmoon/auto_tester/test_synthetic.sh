#for solver in PMGDA EPO MOOSVGD GradHV PMTL GradAgg MGDAUB
##do
##  python libmoon/auto_tester/test_synthetic.py --solver-name $solver --n-epoch 2000
##done
#do
#  python test_synthetic.py --solver-name $solver --n-epoch 20000
#done

nepoch=1
for agg in STche
do
  python test_synthetic.py --solver-name GradAgg --agg-name $agg --n-epoch $nepoch
done

for solver in UMOD
do
  python test_synthetic.py --solver-name $solver --agg-name STche --n-epoch $nepoch
done

for solver in PMGDA EPO MOOSVGD GradHV PMTL MGDAUB
do
  python test_synthetic.py --solver-name $solver --agg-name STche --n-epoch $nepoch
done




sleep 1000






for solver in PMGDA EPO MOOSVGD GradHV PMTL GradAgg MGDAUB
do
  python libmoon/auto_tester/test_mtl.py --solver-name $solver
done

