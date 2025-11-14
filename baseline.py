import os
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import torch.nn.init as init
from torch_geometric.data import Data
import torch

class GATNodeEdgePredictionWithFlowIntensity(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_attr_dim, output_dim, lambda_reg=1, dropout=0.3, heads=4):
        super(GATNodeEdgePredictionWithFlowIntensity, self).__init__()
        # ----------- Node Representations (Three-layer GAT) -----------
        self.gat_conv1 = GATConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            edge_dim=edge_attr_dim,
            heads=heads,
            concat=True,
            dropout=0.0
        )
        self.gat_conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            edge_dim=edge_attr_dim,
            heads=heads,
            concat=True,
            dropout=0.0
        )
        self.gat_conv3 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=output_dim,
            edge_dim=edge_attr_dim,
            heads=1,
            concat=True,
            dropout=0.0
        )

        # ----------- Bilinear Flow Intensity Parameters -----------
        self.M = nn.Parameter(torch.randn(output_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(1))

        # ----------- Regularization Parameter -----------
        self.lambda_reg = lambda_reg

    def forward(self, x, edge_index, edge_attr):
        """
        Parameters:
            x         : [num_nodes, input_dim], Node features
            edge_index: [2, num_edges], Edge indices (PyG format)
            edge_attr : [num_edges, edge_attr_dim], Edge features
        Returns:
            z         : [num_nodes, output_dim], Final node representations
        """
        z1 = self.gat_conv1(x, edge_index, edge_attr)
        z1 = F.leaky_relu(z1, negative_slope=0.01)

        z2 = self.gat_conv2(z1, edge_index, edge_attr)
        z2 = F.leaky_relu(z2, negative_slope=0.01)

        z3 = self.gat_conv3(z2, edge_index, edge_attr)
        z3 = F.leaky_relu(z3, negative_slope=0.01)

        return z3

    def predict_flow_intensity(self, z, edge_index):
        """
        Compute flow intensity f_ij using the bilinear score function.

        Parameters:
            z         : [num_nodes, output_dim], Node embeddings
            edge_index: [2, num_edges], Edge indices

        Returns:
            flow_intensities: [num_edges, 1], Predicted flow intensities
        """
        src, dst = edge_index

        zi = z[src]  # Source node embeddings
        zj = z[dst]  # Destination node embeddings

        # Bilinear score function
        flow_intensities = F.relu(torch.sum(zi @ self.M * zj, dim=-1) + self.b)
        flow_intensities = flow_intensities.unsqueeze(-1)  # [num_edges, 1]

        return flow_intensities

    def loss(self, predicted_flows, true_flows, z):
        """
        Compute loss as MSE loss + node embedding regularization.

        Parameters:
            predicted_flows: [num_edges, 1], Predicted flow intensities
            true_flows     : [num_edges] or [num_edges, 1], True flow intensities
            z              : [num_nodes, output_dim], Node embeddings

        Returns:
            total_loss: Scalar, Total loss
        """
        predicted_flows = predicted_flows.squeeze(-1)  # [num_edges]
        true_flows = true_flows.view(-1)  # [num_edges]

        # MSE loss
        mse_loss = F.mse_loss(predicted_flows, true_flows)

        # Regularization term (L2 norm of node embeddings)
        reg_loss = torch.sum(torch.norm(z, dim=1) ** 2)

        # Total loss
        total_loss = mse_loss + self.lambda_reg * reg_loss
        return total_loss


def set_random_seed(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) 
        torch.cuda.manual_seed_all(seed)  
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 




if __name__ == "__main__":
    seed = 42
    set_random_seed(seed)
    current_date = datetime.now().strftime("%Y%m%d")
    best_model_path = f"best_model/best_gat_model_{current_date}.pth"
    use_best_model_path = f"best_model/best_gat_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load('./data/output/graph_0.005.pth', weights_only=False)

    mean = data.x.mean(dim=0, keepdim=True)
    std = data.x.std(dim=0, keepdim=True) + 1e-9
    data.x = (data.x - mean) / std
    Graphdata = Data(
        x=data.x.to(device),
        edge_index=data.edge_index.to(device),
        edge_attr=data.edge_attr.to(device),  
        edge_true=data.edge_Flow.to(device)   
    )

    train_ratio =1
    num_edges = Graphdata.edge_index.size(1)

    mask = torch.zeros(num_edges, dtype=torch.bool, device=device)
    train_size = int(train_ratio * num_edges)
    mask[:train_size] = True

    train_edge_index = Graphdata.edge_index[:, mask]
    train_edge_attr = Graphdata.edge_attr[mask]
    train_edge_true = Graphdata.edge_true[mask]

    print(f"Train edges: {train_edge_index.size(1)}")
    print(f"Remaining edges (masked out): {num_edges - train_edge_index.size(1)}")

    input_dim = 6
    hidden_dim = 32
    edge_attr_dim = 1
    output_dim = 16

    model = GATNodeEdgePredictionWithFlowIntensity(input_dim, hidden_dim, edge_attr_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 10000  #
    best_val_r2_score = float("-inf")  

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        z_train = model(Graphdata.x, train_edge_index, train_edge_attr)
        # re_edge_weights_train = model.predict_flow_intensity(z_train, train_edge_index, train_edge_attr)
        pre_edge_weights_train = model.predict_flow_intensity(z_train, train_edge_index)

        predicted_weights_train = pre_edge_weights_train.squeeze(-1)

        loss_train = F.mse_loss(predicted_weights_train, train_edge_true)
        loss_train.backward()
        optimizer.step()
        ss_res_train = torch.sum((train_edge_true - predicted_weights_train) ** 2).item()
        ss_tot_train = torch.sum((train_edge_true - torch.mean(train_edge_true)) ** 2).item()
        train_r2_score = 1 - (ss_res_train / ss_tot_train)


        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs}, "
                  f"Training Loss: {loss_train.item():.4f}, Training R²: {train_r2_score:.4f}")


        model.eval()
        with torch.no_grad():

            val_edge_index = Graphdata.edge_index
            val_edge_attr = Graphdata.edge_attr
            val_edge_true = Graphdata.edge_true


            z_val = model(Graphdata.x, val_edge_index, val_edge_attr)
            pre_edge_weights_val = model.predict_flow_intensity(z_val,val_edge_index)

            predicted_weights_val = pre_edge_weights_val.squeeze(-1)

            loss_val = F.mse_loss(predicted_weights_val, val_edge_true)

            ss_res_val = torch.sum((val_edge_true - predicted_weights_val) ** 2).item()
            ss_tot_val = torch.sum((val_edge_true - torch.mean(val_edge_true)) ** 2).item()
            val_r2_score = 1 - (ss_res_val / ss_tot_val)


        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
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
        # losses['val'].append(loss_val.item())
        # r2_scores['val'].append(val_r2_score)

    print("Training complete.")


