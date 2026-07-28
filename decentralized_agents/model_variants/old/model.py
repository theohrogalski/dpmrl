import torch
from torch_geometric.nn.models import GAT
from torch_geometric.nn.conv import GATConv
from torch_geometric.utils import from_networkx,from_nested_tensor
from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
import networkx as nx
from torch_geometric import data
from torch.nn import TransformerEncoder
from torch_geometric.utils import get_laplacian
from torch.nn import MultiheadAttention
from torch_geometric.data import Data
from torch.nn import ReLU
from torch_geometric.nn.aggr import MLPAggregation
from torch_geometric.utils import add_self_loops,to_dense_adj
import gpytorch as gpto
from torch_geometric.nn import TransformerConv

class observation_processing_network(torch.nn.Module):
    def __init__(self,number_of_nodes) :
        if torch.cuda.is_available():
            self.device = "cuda"
            print("using cuda")
        else:
            self.device = "cpu"
        super().__init__()
        self.num_nodes=number_of_nodes
        self.multihead = MultiheadAttention(embed_dim=3, num_heads=3)
        self.number_of_nodes=number_of_nodes
        custom_mlp = torch.nn.Sequential(torch.nn.Linear(5*number_of_nodes,out_features=16),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(16,32),
                                         torch.nn.ReLU(),
                                         torch.nn.Linear(in_features=16,out_features=number_of_nodes)
                                         )
        self.transform_two = TransformerConv(3,3,1)
        self.graph_attention = GAT(in_channels=5,num_layers=10,hidden_channels=3)
        self.actor = MLPAggregation(in_channels=5, out_channels=1,max_num_elements=1,num_layers=3, hidden_channels=5, mlp = custom_mlp)
        self.critic = torch.nn.Linear(in_features=1,out_features=1)
        self.softmax = torch.nn.Softmax()
        self.gp_dict = {}
        self.history=[]
        self.add_laplacian = AddLaplacianEigenvectorPE(49)
        
    def compute_pyg_laplacian_features(self,the_data, k=2):
        """data: PyG Data object for the 50-node graph
    k: number of eigenvectors to return"""
    # 1. Get the Laplacian in COO format (edge_index, edge_weight)
        print(data)
        edge_index, edge_weight = get_laplacian(
            the_data.edge_index, 
            edge_weight=the_data.edge_attr, 
            normalization='sym', 
            num_nodes=50
        )
        
    # 2. Convert to dense for eigenvalue decomposition (safe for 50 nodes)
        L = to_dense_adj(edge_index, max_num_nodes=50).squeeze(0)
        
        # 3. Compute Eigen-decomposition in Torch (GPU compatible!)
        # evals: [50], evecs: [50, 50]
        evals, evecs = torch.linalg.eigh(L)
        
        # 4. Slice the first k eigenvectors (Topology) and the Fiedler Value (Safety)
        lap_ev = evecs[:, :k]  # Shape: [50, k] -> Matches your node features!
        fiedler_val = evals[1] # The 2nd smallest eigenvalue for DCBF
        
        return lap_ev, fiedler_val    
    def forward(self, mental_map:nx.Graph, mask:list):
        mental_map = from_networkx(mental_map, group_node_attrs=["uncertainty","agent_presence","target"])
        print(mental_map.x.shape)
        new_lap = self.compute_pyg_laplacian_features(the_data=mental_map)

# 2. Slice and Overwrite: 
# [:, 3:] selects all rows and columns from index 3 to the end (the 4th and 5th columns)
       # print(new_lap)
        print(mental_map.x.shape)
        if mental_map.x.shape != torch.Size([50,5]):
            print(f"here {mental_map.x.shape}")
            mental_map.x = torch.cat((mental_map.x,torch.zeros(size=([50,2]))),dim=1)
        print("here2")
        print(mental_map.x.shape)
        x_obs = mental_map.x[:, :3]
        x_new = torch.cat((x_obs, new_lap[0]), dim=1) # Join with Laplacian
        # 4. Now pass this fresh [50, 5] tensor to the update function
        ##print(mental_map)
        print("here4")

        
        mental_map.x = mental_map.x.to(dtype=torch.float32)
        mental_map.edge_index = add_self_loops(mental_map.edge_index)
        #print(mental_map.edge_index)
        ##print(type(mental_map))
        gat_x = self.graph_attention(mental_map.x, mental_map.edge_index[0])
        
        self.history.append(gat_x.detach()) 
        
        ##print(f"gat x is {gat_x}")
        
        ##print(self.history)
        
        current_history = torch.concatenate(self.history[:-1] + [gat_x])
        ##print(f"type at 1 is {type(gat_x)}")
        
        results, _ = self.multihead(gat_x, current_history, current_history)
        mental_map.x = results
        ##print(f"type at 2 is {type(results)}")
        ##print(type(results))
        mental_map.edge_index=mental_map.edge_index[0]
        #print(mental_map.edge_index)
        ##print((mental_map).edge_index)
        ##print(mental_map.num_nodes)
        ##print(mental_map.x)
        ##print(mental_map)
        mental_map = self.add_laplacian(mental_map)

        #print(mental_map.edge_index)

        results = self.transform_two(mental_map.x, mental_map.edge_index)
        #print(results)
        ##print((results).shape)
        index=[]
        for i in range(self.number_of_nodes):
            index.append(i)
        index = torch.tensor(index,dtype=torch.int64) 
        ##print(results)   
        print(mental_map.x.shape)
        new_lap = self.compute_pyg_laplacian_features(the_data=mental_map)

        #print(mental_map.edge_index)
        if mental_map.x.shape != torch.Size([50,5]):
            print("here")
            mental_map.x = torch.cat((mental_map.x,torch.zeros(size=([50,2]))),dim=1)
        print("here2")
        print(mental_map.x.shape)
        x_obs = mental_map.x[:, :3]
        x_new = torch.cat((x_obs, new_lap[0]), dim=1) # Join with Laplacian        print("here3")
        

        print(index)
        print(mental_map.x.shape)
        
        results = self.actor(x=mental_map.x, index=index)
        ##print(results)
        #print(type(results))
        
        ##print(type(results))
        ##print(mask)
        
        new_results = results[:,0] * torch.tensor(mask)       
        ##print(f"new results are {new_results}")
        ##print()
        value = self.critic(results)
        """#print(type(new_results))
        #print(new_results)
        #print(new_results.sum())"""
        value = value.mean()
        ##print(value)
        return new_results, value
        

#       Test

if __name__ == "__main__":
    #print("starting process")
    model = observation_processing_network(40)
    graph = nx.read_graphml("./graphs/int_name_graph.graphml")
    model.forward(graph,[0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
