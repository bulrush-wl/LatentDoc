python latentdoc/eval/eval_recon.py \
    --model_type sam_opt_1024_with_ae_with_projector_down4_recon \
    --model_name_or_path exps/recon_test_without_ae_pretrain_aeloss_10/checkpoint-96600 \
    --device cuda \
    --img_path /home/yuhaiyang/zlw/dataset/Vary-600k/imgs/sample_ch.png \
    --output_name output.png