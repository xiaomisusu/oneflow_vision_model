python of_cnn_train_val.py --train_data_dir=data/ofrecord/imagenet/train --train_data_part_num=128 --val_data_dir=data/imagenet/ofrecord/validation --val_data_part_num=128 --num_nodes=1 --gpu_num_per_node=2 --optimizer="sgd" --momentum=0.9 --learning_rate=0.1 --loss_print_every_n_iter=100 --batch_size_per_device=64 --val_batch_size_per_device=50 --num_epoch=120 --model="dla34" 2>&1 | tee ${./train.log}






