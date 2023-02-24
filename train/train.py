import numpy as np
import torch
from model.model import GAT
from data.load_data import read_customers, read_products, create_graph_edges
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected


def load_data():
    customers, customer_mappings = read_customers()
    products, product_mappings = read_products()

    graph_edge_data = create_graph_edges()
    src = [customer_mappings[index] for index in graph_edge_data['customer_unique_id']]
    dst = [product_mappings[index] for index in graph_edge_data['product_id']]
    edge_index = torch.tensor([src, dst])
    edge_attrs = [
        torch.tensor(graph_edge_data[column].values).unsqueeze(dim=1) for column in ['review_score', 'purchase_count']
    ]
    edge_label = torch.cat(edge_attrs, dim=-1)

    review_edge_index = edge_index
    review_edge_label = torch.tensor(graph_edge_data['review_score'].values).unsqueeze(dim=1)

    purchase_edge_index = edge_index
    purchase_edge_label = torch.tensor(graph_edge_data['purchase_count'].values).unsqueeze(dim=1)

    customers_tensor = torch.from_numpy(customers.values)
    products_tensor = torch.from_numpy(products.values)

    data = HeteroData()
    data['customers'].x = customers_tensor
    data['products'].x = products_tensor

    data['customers', 'buys', 'products'].edge_index = purchase_edge_index
    data['customers', 'buys', 'products'].edge_label = purchase_edge_label

    data['customers', 'reviews', 'products'].edge_index = review_edge_index
    data['customers', 'reviews', 'products'].edge_label = review_edge_label

    data = ToUndirected()(data)
    del data['products', 'rev_buys', 'customers'].edge_label
    del data['products', 'rev_reviews', 'products'].edge_label

    return data


if __name__ == '__main__':
    load_data()
