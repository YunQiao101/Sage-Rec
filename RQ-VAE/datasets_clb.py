import numpy as np
import torch
import torch.utils.data as data
import pickle

class EmbDataset(data.Dataset):

    def __init__(self,data_path,data_path_2):

        # self.embeddings = pickle.load(open(data_path_1,'rb')).to('cpu').squeeze().detach().numpy()
        # torch.load(args.cf_emb).squeeze().detach().numpy()
        self.embeddings = np.load(data_path)
        self.clb= torch.load(data_path_2).squeeze().to('cpu').detach().numpy()
        self.dim = self.embeddings.shape[-1]
        self.clb_dim = self.clb.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        clb = self.clb[index]
        return torch.FloatTensor(emb), torch.FloatTensor(clb), index

    def __len__(self):
        return len(self.embeddings)
