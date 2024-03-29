import torch
from torch.nn import Embedding, Linear
from torch_geometric.nn import SAGEConv, GATv2Conv

"""
Model based on https://github.com/pyg-team/pytorch_geometric/blob/master/examples/hetero/bipartite_sage.py
"""


class ItemGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()

        self.conv1 = GATv2Conv(-1, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        return self.lin(x)


class UserGNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, edge_channels):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels)
        self.conv2 = GATv2Conv((-1, -1), hidden_channels, edge_dim=edge_channels, add_self_loops=False)
        self.conv3 = GATv2Conv((-1, -1), hidden_channels, edge_dim=edge_channels, add_self_loops=False)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label):
        product_x = self.conv1(
            x_dict['product'],
            edge_index_dict[('product', 'metapath_0', 'product')],
        ).relu()
        customer_x = self.conv2(
            (x_dict['product'], x_dict['customer']),
            edge_index_dict[('product', 'rev_buys', 'customer')],
            edge_attr=edge_label
            # consider using metapaths for customers
        ).relu()

        customer_x = self.conv3(
            (product_x, customer_x),
            edge_index_dict[('product', 'rev_buys', 'customer')],
            edge_attr=edge_label
        ).relu()

        return self.lin(customer_x)


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
        return z


class MetaGATv2(torch.nn.Module):
    def __init__(self, num_customers, hidden_channels, out_channels, edge_channels=1):
        super().__init__()
        self.customer_emb = Embedding(num_customers, hidden_channels)
        self.customer_encoder = UserGNNEncoder(hidden_channels, out_channels, edge_channels=edge_channels)
        self.item_encoder = ItemGNNEncoder(hidden_channels, out_channels)
        self.decoder = EdgeDecoder(out_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_label=None):
        # x_dict['customer'] = self.customer_emb(x_dict['customer'])
        z_dict = {
            'customer': self.customer_encoder(x_dict, edge_index_dict, edge_label),
            'product': self.item_encoder(
                x_dict['product'],
                edge_index_dict[('product', 'metapath_0', 'product')],
            )}
        return self.decoder(z_dict['customer'], z_dict['product'], edge_label_index)

