for i in 1 2 3 4 5 6 7 8 9 10
do
    for j in 1 2 3 4 5 6 7 8 9 10
    do
        if [ $i -ne $j ]
        then
            python /root/dhk/RobNet/inject.py --cuda --base_class $i --target_class $j \
                                            --save_dir poison_single_patch --save_name single_{$i}_{$j}
        fi
    done
done

for i in 1 2 3 4 5 6 7 8 9 10
do
    for j in 1 2 3 4 5 6 7 8 9 10
    do
        if [ $i -ne $j ]
        then
            python /root/dhk/RobNet/inject.py --cuda --base_class $i --target_class $j \
                                            --save_dir poison_multi_patch --save_name multi_{$i}_{$j} --num_trigger 3
        fi
    done
done