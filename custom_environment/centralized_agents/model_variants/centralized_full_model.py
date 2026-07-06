import torch
import torch.nn as nn
from torch_geometric.nn import GAT, TransformerConv, MLPAggregation
from torch_geometric.utils import from_networkx, get_laplacian, to_dense_adj, add_self_loops
import networkx as nx

class models_full_model(torch.nn.Module):
    
    def get_safe_action_mask(self,mask:list, x_state, neighbors, edge_index, unc_net, threshold=100, eta=0.1,num_moves=0,position=0):
            
            threshold=num_moves
            assert neighbors[position]==1
                # 1. Current Safety h(x_t)
            
            predicted_u_current = unc_net(x_state, edge_index,num_moves)
            h_t = threshold - torch.max(predicted_u_current)
            lower_bound = (1 - eta) * h_t
            safe_mask = torch.zeros(self.number_of_nodes)
            for node_idx in range(self.number_of_nodes):
                h_next = threshold - predicted_u_current[node_idx]
                
                if h_next >= lower_bound:
                    safe_mask[node_idx] = 1
            mask=torch.tensor(mask)*safe_mask
            if mask.sum() == 0:
            min_val=0
            min_node=0
            for node in range(50):
                if node in neighbors:
                    if (threshold - predicted_u_current[node]) < min_val:
                        min_val=threshold - predicted_u_current[node]
                        min_node=node
            mask[min_node] = 1
            mask[position]=1
            return mask
    def __init__(self, number_of_nodes):
        
        super().__init__()
        self.number_of_nodes = number_of_nodes
        if torch.cuda.is_available():
            self.device="cuda"
        else:
            self.device="cpu"
        self.graph_attention = GAT(in_channels=6, hidden_channels=8, num_layers=2, out_channels=5)
        self.multihead = nn.MultiheadAttention(embed_dim=5, num_heads=1)
        self.transform_two = TransformerConv(5, 5, heads=1)
        self.actor = nn.Linear(self.number_of_nodes*5,1)
        self.critic = nn.Linear(in_features=self.number_of_nodes*3, out_features=1)

    def compute_pyg_laplacian_features(self, data, k=2):
        edge_index, _ = get_laplacian(data.edge_index, normalization='sym', num_nodes=self.number_of_nodes)
        L = to_dense_adj(edge_index, max_num_nodes=self.number_of_nodes).squeeze(0)
        evals, evecs = torch.linalg.eigh(L)
        return evecs[:, :k]

    def forward(self, mental_map_nx: nx.Graph, mask: list,unc_net,num_moves,neighbors,position):

        data = from_networkx(mental_map_nx, group_node_attrs=["uncertainty", "agent_presence", "target"])

        lap_ev = self.compute_pyg_laplacian_features(data)
        
        data_x = (data.x).to(self.device).float()
        
        data_x = data_x.flatten()
        
        value = self.critic(data_x)
        
        x_combined = torch.cat([data.x, lap_ev], dim=1) # [50, 5]
        
        uncertainty_prediction = unc_net(data.x, data.edge_index,move_num=num_moves) # [50, 1]
        
        uncertainty_prediction = uncertainty_prediction.to(self.device)
        
        x_combined = x_combined.to(self.device)
        
        x_enriched = torch.cat([x_combined, uncertainty_prediction],1)
        
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=self.number_of_nodes)
        
        x_enriched = x_enriched.to(self.device)
        
        edge_index = edge_index.to(self.device)
        
        x = self.graph_attention(x_enriched, edge_index)
        
        logits = self.actor(x.flatten())
        
        action_mask = self.get_safe_action_mask(x_state=x,edge_index=edge_index,unc_net=unc_net,mask=mask,num_moves=num_moves,neighbors=neighbors,position=position).to(self.device) 
        
        masked_logits = logits  * action_mask
     
        return masked_logits, value, x_combined, edge_index 