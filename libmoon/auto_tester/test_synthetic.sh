nepoch=1000

for agg in STche
do
  python libmoon/auto_tester/test_synthetic.py --solver-name GradAgg --agg-name $agg --n-epoch $nepoch
done

for solver in UMOD
do
  python libmoon/auto_tester/test_synthetic.py --solver-name $solver --agg-name STche --n-epoch $nepoch
done

for solver in PMGDA EPO MOOSVGD GradHV PMTL MGDAUB
do
  python libmoon/auto_tester/test_synthetic.py --solver-name $solver --agg-name STche --n-epoch $nepoch
done

echo over