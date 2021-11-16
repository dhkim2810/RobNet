for i in 0 1 2 3 4 5 6 7 8 9
do
    python /root/dhk/RobNet/inject.py --cuda --base_dir /root/dhk/RobNet --data_dir /root/dataset/CIFAR \
                                    --target_class $i ---trigger_loc 7 \
                                    --save_dir inject/single_single --save_name single_$i
done