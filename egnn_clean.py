from torch import nn
import torch
from torch.nn import Parameter


class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        return normed_coors * self.scale


class E_GCL(nn.Module):
    """
    E(n) Equivariant Convolutional Layer
    re
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False, normalize=False, coords_agg='mean', tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        self.node_norm = nn.LayerNorm(output_nf)
        # self.coors_norm = CoorsNorm()
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.LeakyReLU())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        # agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        if self.coords_agg == 'sum':
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        elif self.coords_agg == 'mean':
            agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        else:
            raise Exception('Wrong coords_agg parameter' % self.coords_agg)
        coord = coord + agg
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)

        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        return h, coord, edge_attr


class EGNN_Model(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0, device='cuda:0', act_fn=nn.SiLU(), n_layers=4, residual=True, attention=False, normalize=False, tanh=False,
                 p=0.6, pool_strategy="topK"):
        super(EGNN_Model, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.pool_strategy = pool_strategy
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf,
                                                act_fn=act_fn, residual=residual, attention=attention,
                                                normalize=normalize, tanh=tanh))
        if pool_strategy == "topK":
            self.pool = Pool(p, hidden_nf, 0.0)
        elif pool_strategy == "GSA":
            self.pool = GSAPool(p, hidden_nf)
        else:
            self.pool = None
        self.unpool = Unpool()
        self.to(self.device)

    def forward(self, h, x, edges, edge_attr=None, G_batch=None):
        for i in range(0, self.n_layers):
            oh = h
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=None)
            if self.pool_strategy == 'GSA':
                sub_h, idx, sub_x, sub_edge_index = self.pool(G_batch, oh, h, x, edges)
                sub_h, sub_x, _ = self._modules["gcl_%d" % i](sub_h, sub_edge_index, sub_x, edge_attr=None)
                sub_h = self.unpool(h.shape[0], sub_h, idx)
                h = torch.max(h, sub_h)
        h = self.embedding_out(h)
        return h, x


def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class Pool(nn.Module):
    def __init__(self, k, in_dim, dropout):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, g_batch, h, coors, edge_index=None):
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()
        scores = self.sigmoid(weights)
        return top_k_graph(g_batch, scores, h, self.k, coors)


class GSAPool(nn.Module):
    def __init__(self, k, in_dim):
        super(GSAPool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.structure = nn.Linear(in_dim, 1)
        self.feature = nn.Linear(in_dim, 1)
        self.weight_var = Parameter(torch.ones(2))

    def forward(self, g_batch, oh, h, x, edges=None):
        weight_var = [torch.exp(self.weight_var[i]) / torch.sum(torch.exp(self.weight_var)) for i in
                      range(2)]
        scores1 = self.sigmoid(self.structure(h).squeeze())
        scores2 = self.sigmoid(self.feature(oh).squeeze())
        scores = weight_var[0] * scores1 + weight_var[1] * scores2
        return top_k_graph(g_batch, scores, h, self.k, x)


class Unpool(nn.Module):
    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, node_nums, h, idx):
        new_h = h.new_zeros([node_nums, h.shape[1]])
        new_h[idx] = h
        return new_h


def top_k_graph(g_batch, scores, h, k, x):
    num_nodes = h.shape[0]
    sub_nodes = max(2, int(k*num_nodes))
    values, idx = torch.topk(scores, sub_nodes)
    new_h = h[idx, :]
    values = torch.unsqueeze(values, -1)
    new_h = torch.mul(new_h, values)
    new_x = x[idx, :]
    sub_graph = g_batch.subgraph(idx)
    sub_edge_index = sub_graph.edges()
    sub_edge_index = torch.stack(sub_edge_index)

    return new_h, idx, new_x, sub_edge_index