import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer2(nn.Module):
    def __init__(self, ndim_in, edims, ndim_out, activation):
        super(GNNLayer2, self).__init__()
        self.W_msg = nn.Linear(ndim_in + edims, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def message_func(self, edges):
        return {'m': F.relu(self.W_msg(torch.cat([edges.src['h'], edges.data['h']], 2)))}

    def forward(self, g_dgl, nfeats, efeats):
        with g_dgl.local_scope():
            g = g_dgl
            g.ndata['h'] = nfeats
            g.edata['h'] = efeats
            g.update_all(self.message_func, fn.sum('m', 'h_neigh'))
            g.ndata['h'] = F.relu(self.W_apply(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 2)))
            return g.ndata['h']


class GCN2(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super(GCN2, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer2(ndim_in, edim, 50, activation))
        self.layers.append(GNNLayer2(50, edim, 25, activation))
        self.layers.append(GNNLayer2(25, edim, ndim_out, activation))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, nfeats, efeats):
        for i, layer in enumerate(self.layers):
            if i != 0:
                nfeats = self.dropout(nfeats)
            nfeats = layer(g, nfeats, efeats)
        return nfeats.sum(1)

if __name__ == '__main__':
    model = GCN2(3, 1, 3, F.relu, 0.5)
    g = dgl.DGLGraph([[0, 2], [2, 3]])
    nfeats = torch.randn((g.number_of_nodes(), 3, 3))
    efeats = torch.randn((g.number_of_edges(), 3, 3, 2))
    model(g, nfeats, efeats)