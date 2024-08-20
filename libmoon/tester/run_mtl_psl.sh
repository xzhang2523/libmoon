for dataset in mnist fashion fmnist
    do
    for device in cpu gpu
    do
      python run_mtl_psl.py --dataset $dataset --device-name $device
    done
done

sleep 100