import torch
import torch.nn as nn
from torch_geometric.nn import GAT, TransformerConv, MLPAggregation
from torch_geometric.utils import from_networkx, get_laplacian, to_dense_adj, add_self_loops
import networkx as nx

class models_no_dcbf_no_state_est(torch.nn.Module):
    def get_safe_action_mask(self,mask, x_state, edge_index, unc_net, threshold=100, eta=0.1,num_moves=0,neighbors=0,position=0):
        """
    Returns a binary mask [50] where 1.0 = Mathematically Safe, 0.0 = Forbidden.
    """
        threshold=num_moves
        assert neighbors[position]==1
        with torch.no_grad():
            # 1. Current Safety h(x_t)
            
            predicted_u_current = unc_net(x_state, edge_index,num_moves)
            h_t = threshold - torch.max(predicted_u_current)

            # 2. Define the Barrier Bound
            # The paper's condition: h(x_t+1) >= (1 - eta) * h(x_t)
            lower_bound = (1 - eta) * h_t
            #print(f"lower bound is {lower_bound}")
            # 3. Predict h(x_t+1) for EVERY possible action (node)
            safe_mask = torch.zeros(self.number_of_nodes)
            for node_idx in range(self.number_of_nodes):
                # Evaluate the safety of the specific target node
                # h(x_t+1) = threshold - uncertainty_at_target_node
                h_next = threshold - predicted_u_current[node_idx]
                
                if h_next >= lower_bound:
                    safe_mask[node_idx] = 1
                    
            # 4. Critical Safety Check (Hard Override)
            # If no nodes are safe (empty set), we MUST pick the node that 
            # maximizes h(x_t+1) to minimize the violation.
        
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
        #print(f"safe mask of cur pos is {mask[position]}")
        mask[position]=1
        return mask
    def __init__(self, number_of_nodes):
        
        super().__init__()
        self.number_of_nodes = number_of_nodes
        if torch.cuda.is_available():
            self.device="cuda"
            ###print("cuda")
        else:
            self.device="cpu"
        # 1. Feature Processing
        self.graph_attention = GAT(in_channels=5, hidden_channels=8, num_layers=2, out_channels=5)
        self.multihead = nn.MultiheadAttention(embed_dim=5, num_heads=1)
        self.transform_two = TransformerConv(5, 5, heads=1)

        # 2. Actor-Critic Heads
        # custom_mlp must match the flattened input of MLPAggregation
        
        self.actor = nn.Linear(self.number_of_nodes*5,1)
        self.critic = nn.Linear(in_features=self.number_of_nodes*3, out_features=1)

    def compute_pyg_laplacian_features(self, the_data, k=2):
        edge_index, _ = get_laplacian(the_data.edge_index, normalization='sym', num_nodes=self.number_of_nodes)
        L = to_dense_adj(edge_index, max_num_nodes=self.number_of_nodes).squeeze(0)
        evals, evecs = torch.linalg.eigh(L)
        return evecs[:, :k], evals[1] # [50, 2], Fiedler Value

    def forward(self, mental_map_nx: nx.Graph, mask: list,unc_net,num_moves,neighbors,position):
        data = from_networkx(mental_map_nx, group_node_attrs=["uncertainty", "agent_presence", "target"])
        # Add Laplacian features [50, 2] to the [50, 3] raw features
        lap_ev, fiedler = self.compute_pyg_laplacian_features(data)
        ###print(lap_ev.shape)
        data_x = (data.x).to(self.device).float()
        
        data_x=data_x.flatten()
        #print(data_x.device)
        value = self.critic(data_x)
        #print(f"value is here {value}")
        
        x_combined = torch.cat([data.x, lap_ev], dim=1) # [50, 5]
        #assert x_combined.shape == torch.Size([50,5])
         # Use no_grad here so Actor doesn't backprop through GCN

        #uncertainty_prediction = unc_net(data.x, data.edge_index,move_num=num_moves) # [50, 1]
    
    # 5. ENRICH THE STATE: Add the prediction as a 6th feature
    # Now state is [50, 6] -> (Obs + Topology + Prediction)
        #uncertainty_prediction = uncertainty_prediction.to(self.device)
        x_combined = x_combined.to(self.device)
        #print(x_combined.shape)
        #print(uncertainty_prediction.shape)
        #x_enriched = torch.cat([x_combined, uncertainty_prediction],1)
        # 2. Graph Processing
        edge_index, _ = add_self_loops(data.edge_index, num_nodes=self.number_of_nodes)
        
        # GAT Layer

        #x_enriched = x_enriched.to(self.device)
        edge_index = edge_index.to(self.device)
        
        x = self.graph_attention(x_combined, edge_index)


        # Attention (Self-attention on the nodes)
        # Multihead expects [Seq, Batch, Embed] -> [50, 1, 5]
        """x_att = x.unsqueeze(1)
        
        attn_out, _ = self.multihead(x_att, x_att, x_att)
        ##print(attn_out.shape)
        x = attn_out.squeeze(1)
        # Transformer Layer
        x = self.transform_two(x, edge_index)"""
        
        ##print(f"x is {x}")
        # 3. Actor-Critic Output
        # index must be [50]
        #idx = torch.arange(self.number_of_nodes, device=self.device)
        
        # MLPAggregation returns [1, number_of_nodes]
        
        logits = self.actor(x.flatten())
      
        return logits, value, x_combined, edge_index 