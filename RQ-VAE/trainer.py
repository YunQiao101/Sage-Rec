import logging
import json
import numpy as np
import torch
import random
from time import time
from torch import optim
from tqdm import tqdm
import swanlab
import torch.nn.functional as F
from utils import ensure_dir,set_color,get_local_time
import os

from datasets_clb import EmbDataset
from torch.utils.data import DataLoader

class Trainer(object):

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.logger = logging.getLogger()

        self.lr = args.lr
        self.learner = args.learner
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_step = min(args.eval_step, self.epochs)
        self.device = args.device
        self.device = torch.device(self.device)
        self.ckpt_dir = args.ckpt_dir
        saved_model_dir = "{}".format(get_local_time())
        self.ckpt_dir = os.path.join(self.ckpt_dir,saved_model_dir)
        ensure_dir(self.ckpt_dir)
        self.labels = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.labels_2 = {"0":[],"1":[],"2":[], "3":[],"4":[], "5":[]}
        self.best_loss = np.inf
        self.best_collision_rate = np.inf
        self.best_loss_ckpt = "best_loss_model.pth"
        self.best_collision_ckpt = "best_collision_model.pth"
        self.optimizer = self._build_optimizer()
        self.model = self.model.to(self.device)
        self.trained_loss = {"total":[],"rqvae":[],"recon":[],"cf":[]}
        self.valid_collision_rate = {"val":[]}
        # 优化：K-means 聚类频率控制
        self.kmeans_interval = getattr(args, 'kmeans_interval', 5)  # 默认每5个epoch执行一次
        self._labels_cached = False  # 标记labels是否已缓存
        self.maxe = args.maxe
        self.use_swanlab = getattr(args, 'use_swanlab', False)

    def _build_optimizer(self):

        params = self.model.parameters()
        learner =  self.learner
        learning_rate = self.lr
        weight_decay = self.weight_decay

        if learner.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate, weight_decay=weight_decay)
        elif learner.lower() == "adagrad":
            optimizer = optim.Adagrad(
                params, lr=learning_rate, weight_decay=weight_decay
            )
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
        elif learner.lower() == "rmsprop":
            optimizer = optim.RMSprop(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        elif learner.lower() == 'adamw':
            # optimizer = optim.AdamW([
            # {'params': self.model.parameters(), 'lr': learning_rate, 'weight_decay':weight_decay}, 
            # {'params': self.awl.parameters(), 'weight_decay':0}
            # ])
            optimizer = optim.AdamW(
                params, lr=learning_rate, weight_decay=weight_decay
            )
        else:
            self.logger.warning(
                "Received unrecognized optimizer, set default Adam optimizer"
            )
            optimizer = optim.Adam(params, lr=learning_rate)
        return optimizer
    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError("Training loss is nan")

    def constrained_km(self, data, n_clusters=10):
        from k_means_constrained import KMeansConstrained 
        # x = data.cpu().detach().numpy()
        # data = self.embedding.weight.cpu().detach().numpy()
        x = data
        size_min = min(len(data) // (n_clusters * 2), 10)
        clf = KMeansConstrained(n_clusters=n_clusters, size_min=size_min, size_max=n_clusters * 6, max_iter=10, n_init=10,
                                n_jobs=10, verbose=False)
        clf.fit(x)
        t_centers = torch.from_numpy(clf.cluster_centers_)
        t_labels = torch.from_numpy(clf.labels_).tolist()

        return t_centers, t_labels
    
    def vq_init(self):
        self.model.eval()
        original_data = EmbDataset(self.args.data_path,self.args.cf_emb_path)
        init_loader = DataLoader(original_data,num_workers=self.args.num_workers,
                             batch_size=len(original_data), shuffle=True,
                             pin_memory=True)
        print(len(init_loader))
        iter_data = tqdm(
                    init_loader,
                    total=len(init_loader),
                    ncols=100,
                    desc=set_color(f"Initialization of vq","pink"),
                    )
        # Train
        for batch_idx, data in enumerate(iter_data):
            data, clb,emb_idx = data[0], data[1],data[2]
            data = data.to(self.device)
            clb = clb.to(self.device)

            self.model.vq_initialization(data,clb)

    def _train_epoch(self, train_data, epoch_idx):

        self.model.train()

        total_loss = 0
        total_recon_loss = 0
        total_cf_loss = 0
        total_quant_loss = 0
        print(len(train_data))
        iter_data = tqdm(
                    train_data,
                    total=len(train_data),
                    ncols=100,
                    desc=set_color(f"Train {epoch_idx}","pink"),
                    )

        # 优化：只在每隔 kmeans_interval 个 epoch 执行 K-means，或在第一个 epoch 执行
        if epoch_idx == 0 or (epoch_idx % self.kmeans_interval == 0):
            print(f"Running K-means clustering for epoch {epoch_idx}")
            embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels[str(idx)] = labels

            embs = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.clb_rq.vq_layers]
            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels_2[str(idx)] = labels
            self._labels_cached = True
        elif not self._labels_cached:
            # 确保 labels 至少初始化一次
            embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels[str(idx)] = labels

            embs = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.clb_rq.vq_layers]
            for idx, emb in enumerate(embs):
                centers, labels = self.constrained_km(emb)
                self.labels_2[str(idx)] = labels
            self._labels_cached = True

        for batch_idx, data in enumerate(iter_data):
            data, clb,emb_idx = data[0], data[1],data[2]
            data ,clb= data.to(self.device),clb.to(self.device)
            self.optimizer.zero_grad()
            out, out_clb,rq_loss, rq_loss_2,indices,indices_2, dense_out,dence_out_2 = self.model(data,clb, self.labels,self.labels_2)
            loss, _, loss_recon, quant_loss = self.model.compute_loss(out, out_clb,rq_loss,rq_loss_2, emb_idx, dense_out,dence_out_2,data,clb)
            self._check_nan(loss)
            loss.backward()
            self.optimizer.step()
            # iter_data.set_postfix_str("Loss: {:.4f}, RQ Loss: {:.4f}".format(loss.item(),rq_loss.item()))
            total_loss += loss.item()
            total_recon_loss += loss_recon.item()
            # total_cf_loss += (cf_loss.item() if cf_loss != 0 else cf_loss)
            total_quant_loss += quant_loss.item()

        # 建议在函数返回前记录平均损失（或者每个 batch 记录一次）
        avg_loss = total_loss / len(train_data)
        avg_recon = total_recon_loss / len(train_data)
        # avg_cf = total_cf_loss / len(train_data)
        avg_quant = total_quant_loss / len(train_data)
        if self.use_swanlab:
            swanlab.log({
                "train/total_loss": avg_loss,
                "train/recon_loss": avg_recon,
                "train/quant_loss": avg_quant,
            }, step=epoch_idx)
        # return total_loss, total_recon_loss, total_cf_loss, quant_loss.item()#这样的话返回的是最后一个batch的量化损失
        return avg_loss, avg_recon,None, avg_quant
    @torch.no_grad()
    def _valid_epoch(self, valid_data):

        self.model.eval()

        iter_data =tqdm(
                valid_data,
                total=len(valid_data),
                ncols=100,
                desc=set_color(f"Evaluate   ", "pink"),
            )
        indices_set = set()

        num_sample = 0
        embs  = [layer.embedding.weight.cpu().detach().numpy() for layer in self.model.rq.vq_layers]
        for idx, emb in enumerate(embs):
            centers, labels = self.constrained_km(emb)
            self.labels[str(idx)] = labels
        for batch_idx, data in enumerate(iter_data):

            data, emb_idx = data[0], data[1]
            num_sample += len(data)
            data = data.to(self.device)
            x_e = self.model.encoder(data)
            _, _, indices = self.model.rq(x_e, self.labels, use_sk=False)
            indices = indices.view(-1,indices.shape[-1]).cpu().numpy()
            for index in indices:
                code = "-".join([str(int(_)) for _ in index])
                indices_set.add(code)

        collision_rate = (num_sample - len(indices_set))/num_sample
        # balance_score = self.balance_overall(tokens_appearance)
        # wandb.log({"collision_rate": collision_rate, "balance_score": 0})
        # 添加 SwanLab 日志
        if self.use_swanlab:
            swanlab.log({
                "eval/collision_rate": collision_rate,
            })
        return collision_rate

    def _save_checkpoint(self, epoch, collision_rate=1, ckpt_file=None):

        ckpt_path = os.path.join(self.ckpt_dir,ckpt_file) if ckpt_file \
            else os.path.join(self.ckpt_dir, 'epoch_%d_collision_%.4f_model.pth' % (epoch, collision_rate))
        state = {
            "args": self.args,
            "epoch": epoch,
            "best_loss": self.best_loss,
            "best_collision_rate": self.best_collision_rate,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path, pickle_protocol=4)

        self.logger.info(
            set_color("Saving current", "blue") + f": {ckpt_path}"
        )

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, loss, recon_loss):
        train_loss_output = (
            set_color("epoch %d training", "green")
            + " ["
            + set_color("time", "blue")
            + ": %.2fs, "
        ) % (epoch_idx, e_time - s_time)
        train_loss_output += set_color("train loss", "blue") + ": %.4f" % loss
        train_loss_output +=", "
        train_loss_output += set_color("reconstruction loss", "blue") + ": %.4f" % recon_loss
        train_loss_output +=", "
        # train_loss_output += set_color("cf loss", "blue") + ": %.4f" % cf_loss
        return train_loss_output + "]"

    def fit(self, data):

        cur_eval_step = 0
        self.vq_init()
        for epoch_idx in range(self.epochs):
            # --- 训练阶段 ---
            training_start_time = time()
            train_loss, train_recon_loss, _, quant_loss = self._train_epoch(data, epoch_idx)
            training_end_time = time()
            # 实时日志输出
            train_loss_output = self._generate_train_loss_output(
                epoch_idx, training_start_time, training_end_time, train_loss, train_recon_loss
            )
            self.logger.info(train_loss_output)

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint(epoch=epoch_idx, ckpt_file=self.best_loss_ckpt)
                # 注意：此处不重置 cur_eval_step，通常早停建议基于验证集指标
                if self.use_swanlab:
                    swanlab.log({"best/train_loss": train_loss}, step=epoch_idx)

            # --- 轻量化验证阶段 ---
            # 吸收代码二：每隔 self.eval_step (如 50 或 100) 进行一次验证
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                collision_rate = self._valid_epoch(data)
                valid_end_time = time()

                valid_score_output = (
                        set_color(f"epoch {epoch_idx} evaluating", "green")
                        + " ["
                        + set_color("time", "blue") + f": {valid_end_time - valid_start_time:.2f}s, "
                        + set_color("collision_rate", "blue") + f": {collision_rate:f}]"
                )
                self.logger.info(valid_score_output)
                if collision_rate < self.best_collision_rate:
                    self.best_collision_rate = collision_rate
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate,
                                          ckpt_file=self.best_collision_ckpt)
                    if self.use_swanlab:
                        swanlab.log({"best/collision_rate": collision_rate}, step=epoch_idx)
                    cur_eval_step = 0  # 找到更好的模型，重置计数器)
                else:
                    cur_eval_step += 1  # 没进步，计数器加 1

                # 执行早停：2000 个验证周期（不是 epoch）无提升则退出
                if cur_eval_step >= self.maxe:
                    self.logger.info(
                        f"Early stopping at epoch {epoch_idx}. Best collision rate: {self.best_collision_rate}")
                    break

                    # 阶段性强制保存（可选，参考原代码逻辑）
                if epoch_idx > 2500:
                    self._save_checkpoint(epoch_idx, collision_rate=collision_rate)
        return self.best_loss, self.best_collision_rate




