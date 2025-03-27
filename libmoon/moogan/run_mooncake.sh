for pref0 in 0.5
do
  python wgan.py --n-epochs 100 --data-name1 moon --data-name2 cake --pref0 $pref0
done

sleep 100


