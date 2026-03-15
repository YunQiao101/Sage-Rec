import argparse
import random
import torch
import numpy as np
from time import time
import logging
# import wandb
import swanlab
import os
# 1. 设置SwanLab日志级别为warning（隐藏debug信息）
os.environ["SWANLAB_LOG_LEVEL"] = "warning"
# 2. 关闭urllib3的debug日志（隐藏HTTP请求日志）
logging.getLogger("urllib3").setLevel(logging.WARNING)
from datasets_clb import EmbDataset
from torch.utils.data import DataLoader
from models.rqvae import RQVAE
from trainer import  Trainer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=2000, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str, default="./data", help="Input data path.")

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mmd", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:4", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--alpha', type=float, default=0.1, help='cf loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='diversity loss weight')
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--sample_strategy', type=str, default="all", help='sample_strategy')
    parser.add_argument('--cf_emb_path', type=str, default="./RQ-VAE/ckpt/Instruments-32d-sasrec.pt", help='cf emb')
   
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128], help='hidden sizes of every layer')
    parser.add_argument("--maxe", type=int, default=2000, help="earlystop")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint", help="output directory for model")
    parser.add_argument('--align', type=float, default=0.01, help='alignment loss coefficient')
    parser.add_argument('--recon', type=float, default=3, help='reconstruction loss coefficient')
    parser.add_argument('--kmeans_interval', type=int, default=5, help='K-means clustering interval in epochs (optimization: reduce CPU-GPU transfer)')
    parser.add_argument('--use_swanlab', action='store_true', help='Enable swanlab logging')
    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 优化：启用 cudnn.benchmark 以提升 GPU 利用率（如果输入尺寸固定）
    # 如果输入尺寸变化较大，设置为 False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    # --- 新增：初始化 SwanLab ---
    enable_swanlab = args.use_swanlab
    if enable_swanlab:
        swanlab.init(
            project="SEGA",  # 项目名称
            experiment_name="01(CF&RQ)",  # 实验名称
            config=vars(args),  # 将 argparse 的参数全部记录
        )
        # --------------------------
    print(args)
    logging.basicConfig(level=logging.DEBUG)
    # cf_emb = torch.load(args.cf_emb).squeeze().detach().numpy()
    data = EmbDataset(args.data_path,args.cf_emb_path)


    """build dataset"""

    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  beta = args.beta,
                  alpha = args.alpha,
                  n_clusters= args.n_clusters,
                  sample_strategy =args.sample_strategy,
                  align = args.align,
                  recon = args.recon,
                  )
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    trainer = Trainer(args,model)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)




