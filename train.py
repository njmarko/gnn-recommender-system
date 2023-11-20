import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

import wandb
from data.load_data import read_customers, read_products, create_graph_edges
from model.bipartite_sage import MetaSage
from model.bipartite_gat import MetaGATv2
from model.model import Model


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    # diff = pred - target.to(pred.dtype)
    # weighted_diff = weight * diff.pow(2)
    # sum_loss = weighted_diff
    # loss = sum_loss.mean()
    loss = (weight * (pred - target.to(pred.dtype)).pow(2)).mean()
    return loss


def load_data(args):
    customers, customer_mappings = read_customers()
    products, product_mappings = read_products()

    graph_edge_data = create_graph_edges()
    src = [customer_mappings[index] for index in graph_edge_data['customer_unique_id']]
    dst = [product_mappings[index] for index in graph_edge_data['product_id']]
    edge_index = torch.tensor([src, dst])
    edge_attrs = [
        torch.tensor(graph_edge_data[column].values).unsqueeze(dim=1) for column in ['review_score', 'purchase_count','timestamp']
    ]
    # If we want all values on the edge
    edge_label = torch.cat(edge_attrs, dim=-1).to(torch.float32)

    review_edge_index = edge_index
    review_edge_label = torch.tensor(graph_edge_data['review_score'].values).unsqueeze(dim=1)

    purchase_edge_index = edge_index
    purchase_edge_label = torch.tensor(graph_edge_data['purchase_count'].values).unsqueeze(dim=1)

    customers_tensor = torch.from_numpy(customers.values).to(torch.float32)
    products_tensor = torch.from_numpy(products.values).to(torch.float32)

    data = HeteroData()
    data['customer'].x = customers_tensor
    data['product'].x = products_tensor

    data['customer', 'buys', 'product'].edge_index = edge_index
    data['customer', 'buys', 'product'].edge_label = edge_label
    # data['customer', 'buys', 'product'].edge_label = review_edge_label

    # data['customer', 'buys', 'product'].edge_index = purchase_edge_index
    # data['customer', 'buys', 'product'].edge_label = purchase_edge_label
    #
    # data['customer', 'reviews', 'product'].edge_index = review_edge_index
    # data['customer', 'reviews', 'product'].edge_label = review_edge_label
    data = ToUndirected()(data)
    # if args.model not in ["meta_sage", "meta_gatv2"]:
        # del data['product', 'rev_buys', 'customer'].edge_label
    # del data['product', 'rev_buys', 'customer'].edge_label

    if args.model in ["meta_sage", "meta_gatv2"]:
        # Generate the co-occurence matrix of movies<>movies:
        metapath = [('product', 'rev_buys', 'customer'), ('customer', 'buys', 'product')]
        data = T.AddMetaPaths(metapaths=[metapath])(data)

        # Apply normalization to filter the metapath:
        _, edge_weight = gcn_norm(
            data['product', 'product'].edge_index,
            num_nodes=data['product'].num_nodes,
            add_self_loops=False,
        )
        edge_index = data['product', 'product'].edge_index[:, edge_weight > 0.002]
        data['product', 'metapath_0', 'product'].edge_index = edge_index

    # del data['product', 'rev_reviews', 'product'].edge_label

    data.validate()
    return data


def split_data(data, val_ratio=0.1, test_ratio=0.1):
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        neg_sampling_ratio=0.0,
        edge_types=[('customer', 'buys', 'product')],
        rev_edge_types=[('product', 'rev_buys', 'customer')],
    )
    return transform(data)


def train(model, data, optimizer, weight=None):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['customer', 'product'].edge_label_index,
                 edge_label=data['product', 'rev_buys', 'customer'].edge_label[:,1:]
                 ).squeeze(axis=-1)
    target = data['customer', 'product'].edge_label[:,0]
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    pred = model(data.x_dict, data.edge_index_dict,
                 data['customer', 'product'].edge_label_index,
                 edge_label=data['product', 'rev_buys', 'customer'].edge_label[:,1:]
                 ).squeeze(axis=-1)
    pred = pred.clamp(min=0, max=5)
    target = data['customer', 'product'].edge_label[:,0].float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


@torch.no_grad()
def top_at_k(model, src, dst, train_data, test_data, k=10):
    customer_idx = random.randint(0, len(src) - 1)
    customer_row = torch.tensor([customer_idx] * len(dst))
    all_product_ids = torch.arange(len(dst))
    edge_label_index = torch.stack([customer_row, all_product_ids], dim=0)
    pred = model(train_data.x_dict, train_data.edge_index_dict,
                 edge_label_index)
    pred = pred.clamp(min=0, max=5)

    # we will only select movies for the user where the predicting rating is =5
    rec_product_ids = (pred[:, 0] == 5).nonzero(as_tuple=True)
    top_k_recommendations = [rec_product for rec_product in rec_product_ids[0].tolist()[:k]]

    test_edge_label_index = test_data['customer', 'product'].edge_label_index
    customer_interacted_products = test_edge_label_index[1, test_edge_label_index[0] == customer_idx]

    hits = 0
    for product_idx in top_k_recommendations:
        if product_idx in customer_interacted_products: hits += 1

    return hits / k


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() and (args.device == 'cuda') else 'cpu'
    wb_run_train = wandb.init(entity=args.entity, project=args.project_name, group=args.group,
                              # save_code=True, # Pycharm complains about duplicate code fragments
                              job_type=args.job_type,
                              tags=args.tags,
                              name=f'{args.model}_train',
                              config=args,
                              )
    graph_data = load_data(args)
    train_data, val_data, test_data = split_data(graph_data)

    # We have an unbalanced dataset with many labels for rating 3 and 4, and very
    # few for 0 and 1, therefore we use a weighted MSE loss.
    if args.use_weighted_loss:
        weight = torch.bincount(train_data['customer', 'product'].edge_label)
        weight = weight.max() / weight
    else:
        weight = None
    if args.model == 'graph_sage':
        model = Model(hidden_channels=32, edge_features=2, metadata=graph_data.metadata())
    elif args.model == 'meta_sage':
        model = MetaSage(train_data['customer'].num_nodes, hidden_channels=64, out_channels=64)
    elif args.model == 'meta_gatv2':
        model = MetaGATv2(train_data['customer'].num_nodes, hidden_channels=args.hidden_channels, out_channels=args.out_channels, edge_channels=args.edge_channels)
    model.to(args.device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        if args.model == 'graph_sage':
            model.encoder(train_data.x_dict, train_data.edge_index_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_model_loss = np.Inf
    best_model_path = None
    for epoch in range(0, args.no_epochs):
        loss = train(model, train_data.to(args.device), optimizer, weight)
        train_rmse = test(model, train_data.to(args.device))
        val_rmse = test(model, val_data.to(args.device))
        if epoch % 10 == 0:
            wb_run_train.log({'train_epoch_loss': loss, 'train_epoch_rmse': train_rmse,
                              'val_epoch_rmse': val_rmse})
        print(f'Epoch: {epoch + 1:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
              f'Val: {val_rmse:.4f}')
        if val_rmse < best_model_loss:
            best_model_loss = val_rmse
            Path(f'../experiments/{args.group}').mkdir(exist_ok=True, parents=True)
            new_best_path = os.path.join(f'../experiments/{args.group}',
                                         f'train-{args.group}-{args.model}-epoch{epoch + 1}'
                                         f'-loss{val_rmse:.4f}.pt')
            torch.save(model.state_dict(), new_best_path)
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = new_best_path
    wb_run_train.finish()

    args.job_type = "eval"
    wb_run_eval = wandb.init(entity=args.entity, project=args.project_name, group=args.group,
                             # save_code=True, # Pycharm complains about duplicate code fragments
                             job_type=args.job_type,
                             tags=args.tags,
                             name=f'{args.model}_eval',
                             config=args,
                             )
    if args.model == 'graph_sage':
        model = Model(hidden_channels=32, edge_features=2, metadata=graph_data.metadata())
    elif args.model == 'meta_sage':
        model = MetaSage(train_data['customer'].num_nodes, hidden_channels=64, out_channels=64)
    elif args.model == 'meta_gatv2':
        model = MetaGATv2(train_data['customer'].num_nodes, hidden_channels=args.hidden_channels, out_channels=args.out_channels, edge_channels=args.edge_channels)
    model.load_state_dict(torch.load(best_model_path))
    model.to(args.device)
    test_rmse = test(model, test_data.to(args.device))
    wb_run_eval.log({'test_rmse': test_rmse})
    wb_run_eval.finish()


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('--use_weighted_loss', action='store_true',
                        help='Whether to use weighted MSE loss.')
    PARSER.add_argument('--no_epochs', default=300, type=int)
    # Wandb logging options
    PARSER.add_argument('-entity', '--entity', type=str, default="weird-ai-yankovic",
                        help="Name of the team. Multiple projects can exist for the same team.")
    PARSER.add_argument('-project_name', '--project_name', type=str, default="gnn-recommender-system",
                        help="Name of the project. Each experiment in the project will be logged separately"
                             " as a group")
    PARSER.add_argument('-group', '--group', type=str, default="default_experiment",
                        help="Name of the experiment group. Each model in the experiment group will be logged "
                             "separately under a different type.")
    PARSER.add_argument('-save_model_wandb', '--save_model_wandb', type=bool, default=True,
                        help="Save best model to wandb run.")
    PARSER.add_argument('-job_type', '--job_type', type=str, default="train",
                        help="Job type {train, eval}.")
    PARSER.add_argument('-tags', '--tags', nargs="*", type=str, default="train",
                        help="Add a list of tags that describe the run.")
    # Model options
    model_choices = ['graph_sage', 'meta_sage', 'meta_gatv2']
    PARSER.add_argument('-m', '--model', type=str.lower, default="meta_gatv2",
                        choices=model_choices,
                        help=f"Model to be used for training {model_choices}")
    PARSER.add_argument('--hidden_channels', default=64, type=int)
    PARSER.add_argument('--out_channels', default=64, type=int)
    PARSER.add_argument('--edge_channels', default=2, type=int)
    # Training options
    PARSER.add_argument('-device', '--device', type=str, default='cuda', help="Device to be used")
    ARGS = PARSER.parse_args()
    main(ARGS)
