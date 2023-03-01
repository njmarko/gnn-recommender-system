import torch
from torch_geometric.nn import SAGEConv
from torch.nn import Embedding, Linear

"""
Model based on https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage.py
"""
class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = SAGEConv(-1, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.conv3 = SAGEConv((-1, -1), hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        movie_x = self.conv1(
            x_dict['item'],
            edge_index_dict[('item', 'metapath_0', 'item')],
        ).relu()

        user_x = self.conv2(
            (x_dict['item'], x_dict['user']),
            edge_index_dict[('item', 'rev_rates', 'user')],
        ).relu()

        user_x = self.conv3(
            (movie_x, user_x),
            edge_index_dict[('item', 'rev_rates', 'user')],
        ).relu()

        return self.lin(user_x)


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_src, z_dst, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_src[row], z_dst[col]], dim=-1)

        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class Model(torch.nn.Module):
    def __init__(self, num_users, hidden_channels, out_channels):
        super().__init__()
        self.user_emb = Embedding(num_users, hidden_channels)
        self.user_encoder = UserGNNEncoder(hidden_channels, out_channels)
        self.movie_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = {}
        x_dict['user'] = self.user_emb(x_dict['user'])
        z_dict['user'] = self.user_encoder(x_dict, edge_index_dict)
        z_dict['item'] = self.movie_encoder(
            x_dict['item'],
            edge_index_dict[('item', 'metapath_0', 'item')],
        )
        return self.decoder(z_dict['user'], z_dict['item'], edge_label_index)
