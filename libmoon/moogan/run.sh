for pref0 in 0.0 0.25 0.5 0.75 1.0
do
  python wgan.py --n-epochs 100 --data-name1 apple --data-name2 mushroom --pref0 $pref0
done




