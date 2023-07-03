# mkdir 'log'
python train.py \
    --data_root '/mnt/nvme1n1p1/kabir/Datasets/ms-celeb-1m-v1c/train_msra/msra_crop' \
    --train_file '/mnt/nvme1n1p1/kabir/Datasets/ms-celeb-1m-v1c/train_msra/train_file.txt' \
    --backbone_type 'MobileFaceNet' \
    --backbone_conf_file '../backbone_conf.yaml' \
    --head_type 'MagFace' \
    --head_conf_file '../head_conf.yaml' \
    --lr 0.1 \
    --out_dir 'out_dir_rerun' \
    --epoches 18 \
    --step '10, 13, 16' \
    --print_freq 200 \
    --save_freq 3000 \
    --batch_size 512 \
    --momentum 0.9 \
    --log_dir 'log_rerun' \
    --tensorboardx_logdir 'mv-hrnet' \
    2>&1 | tee log_rerun/log.log
