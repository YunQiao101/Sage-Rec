import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from info_nce import InfoNCE, info_nce
import random
import collections
from .layers import MLPLayers
from .rq import ResidualVectorQuantizer
from tllib.alignment.dan import MultipleKernelMaximumMeanDiscrepancy
from tllib.modules.kernels import GaussianKernel


class RQVAE(nn.Module):
    def __init__(self,
                 in_dim=768,
                 clb_dim = 32,
                 num_emb_list=None,
                 e_dim=64,
                 layers=None,
                 dropout_prob=0.0,
                 bn=False,
                 loss_type="mse",
                 quant_loss_weight=1.0,
                 kmeans_init=False,
                 kmeans_iters=100,
                 sk_epsilons= None,
                 sk_iters=100,
                 alpha = 1.0,
                 beta = 0.001,
                 n_clusters = 10,
                 sample_strategy = 'all',
                 cf_embedding = 0 ,
                 align=0.1,
                 recon=1
        ):
        super(RQVAE, self).__init__()

        self.in_dim = in_dim
        self.clb_dim = clb_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim
        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight=quant_loss_weight
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.cf_embedding = cf_embedding
        self.alpha = alpha
        self.beta = beta
        self.recon = recon
        self.align=align
        self.n_clusters = n_clusters
        self.sample_strategy = sample_strategy
        # self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(kernels=[GaussianKernel(alpha=2 ** -1)])

        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLPLayers(layers=self.encode_layer_dims,
                                 dropout=self.dropout_prob,bn=self.bn)
        self.clb_encode_layer_dims =[self.clb_dim] + self.layers + [self.e_dim]
        self.clb_encoder = MLPLayers(layers=self.clb_encode_layer_dims,
                                 dropout=self.dropout_prob, bn=self.bn)

        self.rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                          kmeans_init = self.kmeans_init,
                                          kmeans_iters = self.kmeans_iters,
                                          sk_epsilons=self.sk_epsilons,
                                          sk_iters=self.sk_iters,)
        self.clb_rq = ResidualVectorQuantizer(num_emb_list, e_dim, beta=self.beta,
                                              kmeans_init=self.kmeans_init,
                                              kmeans_iters=self.kmeans_iters,
                                              sk_epsilons=self.sk_epsilons,
                                              sk_iters=self.sk_iters,
                                               )
        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLPLayers(layers=self.decode_layer_dims,
                                       dropout=self.dropout_prob,bn=self.bn)
        self.clb_decode_layer_dims = self.clb_encode_layer_dims[::-1]
        self.clb_decode = MLPLayers(layers=self.clb_decode_layer_dims,bn=self.bn)
        self.mkmmd_loss = MultipleKernelMaximumMeanDiscrepancy(kernels=[GaussianKernel(alpha=2 ** -1)])
        self.infonce = InfoNCE()

    def forward(self, x, y,labels, labels_2,use_sk=True):
        x = self.encoder(x)
        y = self.clb_encoder(y)
        x_q, rq_loss, indices = self.rq(x,labels, use_sk=use_sk)
        y_q,rq_loss_2,indices_2 = self.rq(y,labels_2,use_sk=use_sk)
        out = self.decoder(x_q)
        out_clb = self.clb_decode(y)

        # return out, rq_loss, indices, x_q
        return out, out_clb, rq_loss,rq_loss_2,indices, indices_2,x_q,y_q
    
    # def CF_loss(self, quantized_rep, encoded_rep):
    #     batch_size = quantized_rep.size(0)
    #     labels = torch.arange(batch_size, dtype=torch.long, device=quantized_rep.device)
    #     similarities = torch.matmul(quantized_rep, encoded_rep.transpose(0, 1))
    #     cf_loss = F.cross_entropy(similarities, labels)
    #     return cf_loss
    
    def vq_initialization(self,x,y, use_sk=True):
        self.rq.vq_ini(self.encoder(x))
        self.clb_rq.vq_ini(self.clb_encoder(y))

    @torch.no_grad()
    def get_indices(self, xs, ys,labels,labels2, use_sk=False):
        x_e = self.encoder(xs)
        y_e = self.clb_encoder(ys)
        _, _, indices = self.rq(x_e, labels, use_sk=use_sk)
        _, _, indices_2 = self.clb_rq(y_e, labels2, use_sk=use_sk)
        return indices,indices_2

    def compute_loss(self, out, out_clb,quant_loss,quant_loss_2, emb_idx, dense_out, dense_out_2,xs=None,ys=None):

        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')+F.mse_loss(out_clb, ys, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        elif self.loss_type == 'mmd':
            loss_recon = self.mkmmd_loss(out, xs)+self.mkmmd_loss(out_clb, ys)
        else:
            raise ValueError('incompatible loss type')

        # rqvae_n_diversity_loss = loss_recon + self.quant_loss_weight * quant_loss

        # # CF_Loss
        # cf_embedding_in_batch = self.cf_embedding[emb_idx]
        # cf_embedding_in_batch = torch.from_numpy(cf_embedding_in_batch).to(dense_out.device)
        # cf_loss = self.CF_loss(dense_out, cf_embedding_in_batch)

        # total_loss = rqvae_n_diversity_loss + self.alpha * cf_loss
        total_loss = self.recon*loss_recon + self.quant_loss_weight * (quant_loss+quant_loss_2)+self.align*(self.infonce(dense_out,dense_out_2))# 这个是对齐

        return total_loss, None, loss_recon, quant_loss