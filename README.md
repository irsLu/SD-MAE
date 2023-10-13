# Unofficial PyTorch implementation of [Self-distillation Augmented Masked Autoencoders for Histopathological Image Understanding](https://arxiv.org/abs/2203.16983)

<div align="center">
  <img width="100%" alt="LTRP illustration" src="resource/idea.png">
</div>

## Pretrained models
You can choose to download only the weights of the pretrained backbone used for downstream tasks. We also provide the training/evaluation logs. 

<table>
  <tr>
    <th>arch</th>
    <th>params</th>
    <th>mAP</th>
    <th colspan="6">download</th>
  </tr>
  <tr>
    <td>ViT-S/16</td>
    <td>21M</td>
    <td>82.8%</td>
    <td><a href="https://drive.google.com/file/d/1e19tH58NFJYsS6AuWhNwjtBo02-rkYy1/view?usp=sharing">class model only</a></td>
    <td><a href="https://drive.google.com/file/d/1QRwuRyrypeKkSfnH_QVMz6Chyn_q_nMD/view?usp=sharing">args</a></td>
    <td><a href="https://drive.google.com/file/d/1miK3K_bVguKO2P6nXdWhNdPvU-y0xwED/view?usp=sharing">logs</a></td>
    <td><a href="https://drive.google.com/file/d/1miK3K_bVguKO2P6nXdWhNdPvU-y0xwED/view?usp=sharing">coco best</a></td>
    <td><a href="https://drive.google.com/file/d/1tcnPF63LF5g55DLTCwMh1w31W239KOv4/view?usp=sharing">eval logs</a></td>
  </tr>
</table>


## Training
Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.8, PyTorch version 1.12.1, CUDA 11.3 and torchvision 0.13.1. The exact arguments to reproduce the models presented in our paper can be found in the `args` column of the [pretrained models section](https://drive.google.com/file/d/1QRwuRyrypeKkSfnH_QVMz6Chyn_q_nMD/view?usp=sharing). 



###  Single-node training
Run LTRP with ViT-small classing model on a single node with 8 GPUs for 400 epochs with the following command. We provide [training](https://drive.google.com/file/d/1miK3K_bVguKO2P6nXdWhNdPvU-y0xwED/view?usp=sharing) logs for this run to help reproducibility.
```
python -m torch.distributed.launch --nproc_per_node=8  ltrp/main_ltrp.py  \
    --data_path yourpath/imagenet_2012 \
    --batch_size 512 \
    --model ltrp_base_and_vs \
    --mask_ratio 0.9 \
    --epochs 400 \
    --resume_from_mae yourpath/mae_visualize_vit_base.pth \
    --ltr_loss list_mleEx \
    --list_mle_k 20 \
    --asymmetric
```

###  Single-node finetuning on MC-COCO dataset
```
python -m torch.distributed.launch --nproc_per_node=4  ltrp/main_ml.py \
    --finetune_ltrp yourpath/pretrained_ckpt.pth \
    --finetune yourpath/mae_pretrain_vit_base.pth \
    --data_path  yourpath/coco2017/ \
    --batch_size 256 \
    --decoder_embedding 768 \
    --epochs 100 \
    --dist_eval \
    --score_net ltrp_cluster \
    --keep_nums 147 \
    --nb_classes 80 \
    --ltrp_cluster_ratio 0.7
```


