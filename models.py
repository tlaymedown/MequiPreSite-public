
import numpy as np
from config import Config
from egnn_clean import *
from vnegnn_offical import *

import warnings
warnings.filterwarnings("ignore")

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SoftWeighted(nn.Module):
    def __init__(self, num_view):
        super(SoftWeighted, self).__init__()
        self.num_view = num_view
        self.weight_var = Parameter(torch.ones(num_view))

    def forward(self, data):
        weight_var = [torch.exp(self.weight_var[i]) / torch.sum(torch.exp(self.weight_var)) for i in
                      range(self.num_view)]
        final_pred = 0
        for i in range(self.num_view):
            final_pred += weight_var[i] * data[i]
        return final_pred, weight_var


class EGNNPPIS(nn.Module):
    def __init__(self, in_dim=67, in_edge_dim=2, hidden_dim=67, out_dim=2, layers=10, attention=True, device='cuda:0', tanh=False, normalize=False, p=0.6, pool_strategy="topK"):
        super(EGNNPPIS, self).__init__()
        self.model = EGNN_Model(in_node_nf=in_dim, in_edge_nf=in_edge_dim, hidden_nf=hidden_dim, n_layers=layers,
                                attention=attention,out_node_nf=out_dim, tanh=tanh, normalize=normalize, p=p, pool_strategy=pool_strategy)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10,
                                                                    min_lr=1e-6)

    def forward(self, n_feats, edge_index, pos, e_feats=None, G_batch=None):
        edge_index = edge_index.squeeze(0)
        n_feats = n_feats.squeeze(0)
        h, c = self.model(n_feats, pos, edge_index, e_feats, G_batch=G_batch)
        return h


class EGNNPPIS2(nn.Module):
    def __init__(self, in_dim=3328, in_edge_dim=2, hidden_dim=128, out_dim=2, layers=10, attention=True, device='cuda:0', tanh=False, normalize=False, p=0.6, pool_strategy="topK"):
        super(EGNNPPIS2, self).__init__()
        self.model = EGNN_Model(in_node_nf=hidden_dim, in_edge_nf=in_edge_dim, hidden_nf=hidden_dim, n_layers=layers,
                                attention=attention,out_node_nf=out_dim, tanh=tanh, normalize=normalize, p=p, pool_strategy=pool_strategy)
        self.embedding_in = nn.Linear(in_dim, hidden_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=10,
                                                                    min_lr=1e-6)

    def forward(self, n_feats, edge_index, pos, e_feats=None, G_batch=None):
        edge_index = edge_index.squeeze(0)
        n_feats = n_feats.squeeze(0)
        n_feats = self.embedding_in(n_feats)
        h, c = self.model(n_feats, pos, edge_index, e_feats, G_batch=G_batch)
        return h


class VNEGNNPPIS(nn.Module):
    def __init__(self, in_dim=67, in_edge_dim=0, hidden_dim=67, layers=6):
        super(VNEGNNPPIS, self).__init__()
        self.model = EGNNGlobalNodeHetero(node_features=in_dim, edge_features=in_edge_dim, hidden_features=hidden_dim,
                                          num_layers=layers,
                                          out_features=hidden_dim, dropout=0.0, weight_share=False)
        self.out = nn.Linear(hidden_dim, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=5,
                                                                    min_lr=1e-6)

    def forward(self, node_feat, node_pos, virtual_node_feat, virtual_node_pos, edge_index, A2V_edge_index, V2A_edge_index):
        edge_index = edge_index.squeeze(0)
        node_feat = node_feat.squeeze(0)
        virtual_node_feat = virtual_node_feat.squeeze(1)
        A2V_edge_index = A2V_edge_index.squeeze(0)
        V2A_edge_index = V2A_edge_index.squeeze(0)

        h, x_global_node, pos_atom, pos_global_node = self.model(node_feat, node_pos, virtual_node_feat,
                                                                 virtual_node_pos, edge_index, A2V_edge_index,
                                                                 V2A_edge_index)
        out = self.out(h)
        return out


class VNEGNNPPIS2(nn.Module):
    def __init__(self, in_dim=3328, in_edge_dim=0, hidden_dim=67, layers=6):
        super(VNEGNNPPIS2, self).__init__()
        self.embedding_in = nn.Linear(in_dim, hidden_dim)
        self.model = EGNNGlobalNodeHetero(node_features=hidden_dim, edge_features=in_edge_dim, hidden_features=hidden_dim,
                                          num_layers=layers,
                                          out_features=hidden_dim, dropout=0.0, weight_share=False)
        self.out = nn.Linear(hidden_dim, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.6, patience=5,
                                                                    min_lr=1e-6)

    def forward(self, node_feat, node_pos, virtual_node_feat, virtual_node_pos, edge_index, A2V_edge_index, V2A_edge_index):
        edge_index = edge_index.squeeze(0)
        node_feat = node_feat.squeeze(0)
        virtual_node_feat = virtual_node_feat.squeeze(1)
        A2V_edge_index = A2V_edge_index.squeeze(0)
        V2A_edge_index = V2A_edge_index.squeeze(0)
        node_feat = self.embedding_in(node_feat)
        virtual_node_feat = self.embedding_in(virtual_node_feat)
        h, x_global_node, pos_atom, pos_global_node = self.model(node_feat, node_pos, virtual_node_feat,
                                                                 virtual_node_pos, edge_index, A2V_edge_index,
                                                                 V2A_edge_index)
        h = self.out(h)
        return h