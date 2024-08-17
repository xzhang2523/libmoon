seed_num=1
for seed in {1,...,$seed_num}
  do
  for mtd in mgda tche mtche
  do
    python run_mtl_clean.py --mtd mtd --seed $seed
  done
done
sleep 100