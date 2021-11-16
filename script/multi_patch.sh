for i in 0 1 2 3 4 5 6 7 8 9
do
    python /root/dhk/RobNet/single_inject.py --cuda --base_dir /root/dhk/RobNet --data_dir /root/dataset/CIFAR \
                                    --target_class $i ---trigger_loc [3, 7] --num_trigger 2 \
                                    --save_dir inject/single_multi --save_name single_$i
done