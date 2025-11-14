import random
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class GATNodeEdgePrediction(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_attr_dim, output_dim, lambda_reg=1, beta=2.0, dropout=0.3):
        super(GATNodeEdgePrediction, self).__init__()
        self.gat_conv1 = GATConv(input_dim, hidden_dim, edge_dim=edge_attr_dim, heads=4, concat=True)
        self.gat_conv2 = GATConv(hidden_dim*4 , hidden_dim, edge_dim=edge_attr_dim, heads=4, concat=True)
        self.gat_conv3 = GATConv(hidden_dim*4, output_dim, edge_dim=edge_attr_dim, heads=1, concat=True)

        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32), requires_grad=True)

        self.residual1 = nn.Linear(input_dim, hidden_dim*4 )
        self.residual2 = nn.Linear(hidden_dim*4 , hidden_dim*4)
        self.residual3 = nn.Linear(hidden_dim*4 , output_dim)

        self.node_ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        edge_emb_dim = 16
        edge_mlp_input_dim = output_dim * 3 + edge_attr_dim + input_dim * 3

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_mlp_input_dim, 128),
            # nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Dropout(0.3),  
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            ScaledSoftplus(beta=2.0)
        )
        self.dropout = nn.Dropout(dropout) 
        self.lambda_reg = lambda_reg

        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_attr_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

    def forward(self, x, edge_index, edge_attr):

        x = x * self.feature_attention(x)
        x_residual1 = self.residual1(x)
        z1 = self.gat_conv1(x, edge_index, edge_attr)
        z1 = self.dropout(F.leaky_relu(z1, negative_slope=0.01))  # Dropout after activation
        z1 = z1 + x_residual1

        z2 = self.gat_conv2(z1, edge_index, edge_attr)
        z2 = self.dropout(F.leaky_relu(z2, negative_slope=0.01))  # Dropout after activation
        x_residual2 = self.residual2(z1)  
        z2 = z2 + x_residual2

        # 第三层 GAT
        z3 = self.gat_conv3(z2, edge_index, edge_attr)
        z3 = self.dropout(F.leaky_relu(z3, negative_slope=0.01))  # Dropout after activation
        z_self = self.node_ffn(x)
        z3 = z3 + z_self
        return z3



    def predict_edge_weight(self, z, edge_index, edge_attr, x):
        src, dst = edge_index
        zi = F.normalize(z[src], p=2, dim=-1)
        zj = F.normalize(z[dst], p=2, dim=-1)
        edge_attr = edge_attr.pow(-self.beta).unsqueeze(-1)

        xi = x[src]
        xj = x[dst]

        edge_features = torch.cat([
            zi, zj, zi - zj,
            edge_attr,
            xi, xj, xi - xj
        ], dim=-1)

        edge_weights = self.edge_mlp(edge_features)
        return edge_weights



    def loss(self, predicted_weights, true_weights, z):
        predicted_weights = predicted_weights.squeeze(-1)  
        true_weights = true_weights.view(-1) 

        mse_loss = F.mse_loss(predicted_weights, true_weights)

        reg_loss = torch.sum(torch.norm(z, dim=1) ** 2)
        total_loss = mse_loss + self.lambda_reg * reg_loss
        return total_loss

class ScaledSoftplus(nn.Module):
    def __init__(self, beta=3.0):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return self.beta * F.softplus(x)

def weighted_mse(pred, target):
    weights = torch.where(target < 5, 2.0, 1.0)
    return torch.mean(weights * (pred - target) ** 2)

def set_random_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False  

def simulate_scenario(original_data, scenario):

    modified_data = original_data.clone()

    if 'node_features' in scenario:
        for node_idx, new_features in scenario['node_features'].items():
            modified_data.x[node_idx] = torch.tensor(new_features, dtype=modified_data.x.dtype, device=modified_data.x.device)

    if 'edge_attributes' in scenario:
        for edge_idx, new_attr in scenario['edge_attributes'].items():
            modified_data.edge_attr[edge_idx] = torch.tensor(new_attr, dtype=modified_data.edge_attr.dtype, device=modified_data.edge_attr.device)

    if 'add_edges' in scenario:
        add_edge_index = torch.tensor(scenario['add_edges'], dtype=torch.long, device=modified_data.edge_index.device).t().contiguous()
        add_edge_attr = torch.tensor([scenario['new_edge_attr']] * len(scenario['add_edges']), dtype=modified_data.edge_attr.dtype, device=modified_data.edge_attr.device).unsqueeze(-1)
        modified_data.edge_index = torch.cat([modified_data.edge_index, add_edge_index], dim=1)
        modified_data.edge_attr = torch.cat([modified_data.edge_attr, add_edge_attr], dim=0)

    if 'remove_edges' in scenario:
        mask = torch.ones(modified_data.edge_index.size(1), dtype=torch.bool, device=modified_data.edge_index.device)
        mask[scenario['remove_edges']] = False
        modified_data.edge_index = modified_data.edge_index[:, mask]
        modified_data.edge_attr = modified_data.edge_attr[mask]

    return modified_data


def what_if_analysis(model, original_data, scenarios, device):
    model.eval()
    results = []

    for idx, scenario in enumerate(scenarios):
        modified_data = simulate_scenario(original_data, scenario)
        modified_data = modified_data.to(device)

        with torch.no_grad():
            z = model(modified_data.x, modified_data.edge_index, modified_data.edge_attr)
            predicted_weights = model.predict_edge_weight(z, modified_data.edge_index, modified_data.edge_attr,Graphdata.x).squeeze(-1)
            results.append({
                'scenario_id': idx,
                'predicted_edge_weights': predicted_weights.cpu().numpy()
            })

    return results


def visualize_predictions_withbox(actual_tensor, predicted_tensor):

    predicted = predicted_tensor.detach().cpu().numpy()
    actual = actual_tensor.detach().cpu().numpy()

    # predicted = np.log(predicted)
    # actual = np.log(actual)
    plt.figure(figsize=(10, 8))


    plt.scatter(actual, predicted, alpha=0.6, color='gray', edgecolor='none')


    max_val = max(max(actual), max(predicted))
    min_val = min(min(actual), min(predicted))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y = x')


    data = pd.DataFrame({'Actual': actual, 'Predicted': predicted})
    bins = np.linspace(min_val, max_val, 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    data['bin'] = pd.cut(data['Actual'], bins=bins, labels=False, include_lowest=True)

    for bin_idx in sorted(data['bin'].dropna().unique()):
        bin_values = data.loc[data['bin'] == bin_idx, 'Predicted']
        if len(bin_values) > 0:
            q9 = np.percentile(bin_values, 9)
            q91 = np.percentile(bin_values, 91)
            color = "green" if q9 <= bin_centers[bin_idx] <= q91 else "red"


            plt.boxplot(
                bin_values,
                positions=[bin_centers[bin_idx]],
                widths=0.4,
                patch_artist=True,
                boxprops=dict(facecolor=color, color=color, alpha=0.6),
                medianprops=dict(color="black"),
                showfliers=False
            )


            mean_value = np.mean(bin_values)  
            plt.scatter(
                bin_centers[bin_idx],
                mean_value,
                color="black",
                s=50,
                zorder=3,
                label="Mean" if bin_idx == 0 else None
            )


    plt.xticks(bin_centers, [f"{x:.2f}" for x in bin_centers], rotation=45, fontsize=12)


    # plt.title(f"{model_name}: Predicted vs Actual with Conditional Boxplots", fontsize=16, fontweight='bold')
    plt.xlabel("Actual Flow", fontsize=18)
    plt.ylabel("Predicted Flow", fontsize=18)


    plt.legend(fontsize=18, loc='upper left')
    plt.grid(True)


    plt.tight_layout()
    plt.savefig(f"./figure/EGFN_among_cities_with_boxplots.png", dpi=300)


    plt.show()

if __name__ == "__main__":

    seed = 42
    set_random_seed(seed)
    current_date = datetime.now().strftime("%Y%m%d")

    best_model_path = f"best_model/graph_0.5_best_model.pth"
    use_best_model_path = f"best_model/graph_0.5_best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    data = torch.load('./Construsted Graph/graph_0.5.pth', weights_only=False)


    mean = data.x.mean(dim=0, keepdim=True)
    std = data.x.std(dim=0, keepdim=True) + 1e-9
    data.x = (data.x - mean) / std


    Graphdata = Data(
        x=data.x.to(device),
        edge_index=data.edge_index.to(device),
        edge_attr=data.edge_attr.to(device),  
        edge_true=data.edge_Flow.to(device)   
    )

    train_ratio =0.9
    num_edges = Graphdata.edge_index.size(1)
    perm = torch.randperm(num_edges, device=device)
    train_size = int(train_ratio * num_edges)
    train_idx = perm[:train_size]
    mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    mask[train_idx] = True

    train_edge_index = Graphdata.edge_index[:, mask]
    train_edge_attr = Graphdata.edge_attr[mask]
    train_edge_true = Graphdata.edge_true[mask]
    train_nodes = torch.unique(train_edge_index)
    train_x_attr = Graphdata.x[train_nodes]


    print(f"Train edges: {train_edge_index.size(1)}")
    print(f"Remaining edges (masked out): {num_edges - train_edge_index.size(1)}")

    input_dim = 6
    hidden_dim = 128
    edge_attr_dim = 1
    output_dim = 16

    model = GATNodeEdgePrediction(input_dim, hidden_dim, edge_attr_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 30000
    best_val_r2_score = float("-inf")  
    losses = {'train': [], 'val': []}
    r2_scores = {'train': [], 'val': []}

    for epoch in range(num_epochs):

        model.train()
        optimizer.zero_grad()

        z_train = model(Graphdata.x, train_edge_index, train_edge_attr)
        pre_edge_weights_train = model.predict_edge_weight(z_train, train_edge_index, train_edge_attr,train_x_attr)

        predicted_weights_train = pre_edge_weights_train.squeeze(-1)

        loss_train = F.mse_loss(predicted_weights_train, train_edge_true)
        loss_train.backward()
        optimizer.step()

        rmse_train = torch.sqrt(torch.mean((train_edge_true - predicted_weights_train) ** 2))

        ss_res_train = torch.sum((train_edge_true - predicted_weights_train) ** 2).item()
        ss_tot_train = torch.sum((train_edge_true - torch.mean(train_edge_true)) ** 2).item()
        train_r2_score = 1 - (ss_res_train / ss_tot_train)


        if (epoch + 1) % 1000 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Training Loss: {loss_train.item():.4f}, Training R²: {train_r2_score:.4f}")

        losses['train'].append(loss_train.item())
        r2_scores['train'].append(train_r2_score)


        model.eval()
        with torch.no_grad():

            val_edge_index = Graphdata.edge_index
            val_edge_attr = Graphdata.edge_attr
            val_edge_true = Graphdata.edge_true


            z_val = model(Graphdata.x, val_edge_index, val_edge_attr)
            pre_edge_weights_val = model.predict_edge_weight(z_val, val_edge_index, val_edge_attr,Graphdata.x)


            predicted_weights_val = pre_edge_weights_val.squeeze(-1)


            loss_val = F.mse_loss(predicted_weights_val, val_edge_true)


            ss_res_val = torch.sum((val_edge_true - predicted_weights_val) ** 2).item()
            ss_tot_val = torch.sum((val_edge_true - torch.mean(val_edge_true)) ** 2).item()
            val_r2_score = 1 - (ss_res_val / ss_tot_val)


        if (epoch + 1) % 1000 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Validation Loss: {loss_val.item():.4f}, Validation R²: {val_r2_score:.4f}")
            predicted_edge_weights_np = predicted_weights_val.cpu().detach().numpy()  
            val_edge_true_np = val_edge_true.cpu().detach().numpy()  

            print(f"Epoch {epoch + 1}/{num_epochs}")
            print("Predicted edge weights :", predicted_edge_weights_np)
            print("True edge weights :", val_edge_true_np)
        if val_r2_score > best_val_r2_score:
            best_val_r2_score = val_r2_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_r2_score': best_val_r2_score,
            }, best_model_path)
            print(f"Epoch {epoch + 1}: Best model saved with Validation R²: {val_r2_score:.4f}")
        losses['val'].append(loss_val.item())
        r2_scores['val'].append(val_r2_score)


    print("Training complete.")


    print("Loading the best model for evaluation...")
    checkpoint = torch.load(use_best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    best_epoch = checkpoint.get('epoch', 'Unknown') 
    best_val_r2_score = checkpoint.get('best_val_r2_score', 'Unknown')  

    print(f"Best model loaded from epoch {best_epoch} with Validation R²: {best_val_r2_score:.4f}")

    z = model(Graphdata.x, Graphdata.edge_index, Graphdata.edge_attr)
    predicted_weights = model.predict_edge_weight(z, Graphdata.edge_index, Graphdata.edge_attr,Graphdata.x).squeeze(-1)
    visualize_predictions_withbox(predicted_weights, Graphdata.edge_true)


    def modify_features(x, feature_indices, factor):
        x_mod = x.clone()
        x_mod[:, feature_indices] *= factor
        return x_mod


    scenarios = [
        #scenarios0
        {
            'node_features': {
                i: (lambda x: (x[0:2].mul_(1.5), x)[1])(Graphdata.x[i].clone()).tolist()
                for i in range(Graphdata.x.size(0))

            },
        },
        # scenarios1
        {
            'node_features': {
                               i: (lambda x: (x[2:4].mul_(1.5), x)[1])(Graphdata.x[i].clone()).tolist()
                for i in range(Graphdata.x.size(0))

            },
        },
        # scenarios2
        {
            'node_features': {
                  i: (lambda x: (x[4:6].mul_(1.5), x)[1])(Graphdata.x[i].clone()).tolist()
                for i in range(Graphdata.x.size(0))
            },
        },
        # scenarios3
        {
            'node_features': {
                i: (Graphdata.x[i] * 1.5).tolist() for i in range(len(Graphdata.x))  
            },
        },


        # scenarios4
        {   'node_features': {
                i: (Graphdata.x[i] * 1.5).tolist() for i in range(len(Graphdata.x))  
            },
            'edge_attributes': {
                i: (Graphdata.edge_attr[i] * 0.8).tolist() for i in range(Graphdata.edge_attr.size(0))
            }
        },
        # scenarios5
        {
            'node_features': {
                i: (Graphdata.x[i] * 1).tolist() for i in range(6)
            },
        },
    ]
    #

    # 执行 What-if 分析
    analysis_results = what_if_analysis(model, Graphdata, scenarios, device)

    # 处理和可视化结果
    for result in analysis_results:
        scenario_id = result['scenario_id']
        predicted_weights = result['predicted_edge_weights']

        edge_index = Graphdata.edge_index.cpu().numpy()
        # tensor2_np = predicted_weights.cpu().numpy()
        print(edge_index)
        print(predicted_weights)

        combined = np.column_stack((edge_index.T, predicted_weights))

        df_combined = pd.DataFrame(combined, columns=['origin','destination', 'Flow'])
        # df_combined.to_csv(f'./simulation_2/scenario_{scenario_id}.csv', index=False)


        # print(f"Scenario {scenario_id}:")
        # print(Graphdata.edge_index)
        # print(predicted_weights)


