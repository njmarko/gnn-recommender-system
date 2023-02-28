import argparse
import torch

import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit, ToUndirected

from model.model import Model
from data.load_data import read_customers, read_products, create_graph_edges
import wandb


def weighted_mse_loss(pred, target, weight=None):
    weight = 1. if weight is None else weight[target].to(pred.dtype)
    diff = pred - target.to(pred.dtype)
    weighted_diff = weight * diff
    sum_loss = weighted_diff.pow(2)
    loss = sum_loss.mean()
    # loss = (weight * (pred - target.to(pred.dtype)).pow(2)).mean()
    return loss


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

    # data['customer', 'buys', 'product'].edge_index = purchase_edge_index
    # data['customer', 'buys', 'product'].edge_label = purchase_edge_label
    #
    # data['customer', 'reviews', 'product'].edge_index = review_edge_index
    # data['customer', 'reviews', 'product'].edge_label = review_edge_label

    data = ToUndirected()(data)
    del data['product', 'rev_buys', 'customer'].edge_label
    # del data['product', 'rev_reviews', 'product'].edge_label

    return data


def split_data(data, val_ratio=0.1, test_ratio=0.1):
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        neg_sampling_ratio=0.0,
        edge_types=[('customer', 'buys', 'product')],
        rev_edge_types=[('product', 'rev_buys', 'customer')],
    )
    return transform(data)


def train(model, data, optimizer, weight=None):
    model.train()
    optimizer.zero_grad()
    pred = model(data.x_dict, data.edge_index_dict,
                 data['customer', 'product'].edge_label_index)
    target = data['customer', 'product'].edge_label
    loss = weighted_mse_loss(pred, target, weight)
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data):
    pred = model(data.x_dict, data.edge_index_dict,
                 data['customer', 'product'].edge_label_index)
    pred = pred.clamp(min=0, max=5)
    target = data['customer', 'product'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse)


def main(args):
    wb_run_train = wandb.init(entity=args.entity, project=args.project_name, group=args.group,
                              # save_code=True, # Pycharm complains about duplicate code fragments
                              job_type=args.job_type,
                              tags=args.tags,
                              name=f'{args.model}_train',
                              config=args,
                              )
    graph_data = load_data()
    train_data, val_data, test_data = split_data(graph_data)

    # We have an unbalanced dataset with many labels for rating 3 and 4, and very
    # few for 0 and 1, therefore we use a weighted MSE loss.
    if args.use_weighted_loss:
        weight = torch.bincount(train_data['customer', 'product'].edge_label)
        weight = weight.max() / weight
    else:
        weight = None

    model = Model(hidden_channels=32, edge_features=2, metadata=graph_data.metadata())

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, args.no_epochs + 1):
        loss = train(model, train_data, optimizer, weight)
        train_rmse = test(model, train_data)
        val_rmse = test(model, val_data)
        test_rmse = test(model, test_data)
        wb_run_train.log({'train_epoch_loss': loss, 'val_epoch_loss': val_rmse})
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
              f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')
    wb_run_train.finish()


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
    model_choices = ['GRAPH_SAGE']
    PARSER.add_argument('-m', '--model', type=str.lower, default="GRAPH_SAGE",
                        choices=model_choices,
                        help=f"Model to be used for training {model_choices}")
    ARGS = PARSER.parse_args()
    main(ARGS)
