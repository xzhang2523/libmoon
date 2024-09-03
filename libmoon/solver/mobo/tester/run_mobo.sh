FE=200
n_var=8
for problem in RE37
  do
  for seed in {0..4}
  do
    python run_dirhvego.py --seed $seed --problem-name $problem --FE $FE --use-fig False --n-var $n_var
    python run_psl_dirhvei.py --seed $seed --problem-name $problem --FE $FE --use-fig False --n-var $n_var
    python run_pslmobo.py --seed $seed --problem-name $problem --FE $FE --use-fig False --n-var $n_var
  done
done
