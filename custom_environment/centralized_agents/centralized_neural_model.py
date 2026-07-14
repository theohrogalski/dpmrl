import torch
from torch.nn import Module
import networkx as nx
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
class uncertainty_estimator(Module):
    """_This module is the same as the decentralized neural state estimator, except that it
    takes in the observations from every agent, not just one._
    """
    def __init__(self, feature_dim:int, hidden_dim:int, out_dim:int, num_nodes:int, num_agents:int,max_moves:int):
        super().__init__()
        
        if torch.cuda.is_available():
            print("cuda2")
            self.device="cuda"
        else: 
            self.device="cpu"
        self.episodes=0
        self.num_nodes=num_nodes
        self.max_moves=max_moves
            # optional second layer for better graph depth                  
        self.lin = torch.nn.utils.spectral_norm(torch.nn.Linear(2,1)).to(self.device)
        self.loss_data=[]
        self.gamma =0.99
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        self.loss_f = torch.nn.MSELoss()
        self.current_loss=0

    def forward(self,x,edge_index,move_num):
        #print(type(edge_index))
        x= x.to(self.device)
        x=x[:,0]
        move_num = torch.tensor(move_num).expand(self.num_nodes)
        move_num = move_num.to(self.device).float()
        #print(move_num.shape)
        #print(x.shape)
        x = torch.concat((x.unsqueeze(1),move_num.unsqueeze(1)),1).to(self.device)

        x=(self.lin(x))*self.gamma
        #print(f"shape of x here iss {x.shape}")
        return x
    
    def update_estimator(self, x, edge_index,move_num):
        """_This function takes updates the estimation model via taking in graph data, 
        using it to run a forward pass of the model, and comparing the result to the 
        actual data. Essentially: error = |G|^-1 * sum((x_t-f(x)_t)^n), where x is the graph data
        , f(x) is the neural model and |G| is the number of nodes in the graph. The data is augmented
        with the current move number allowing for temporal relations to be learned. 

        Args:
            x (_graph data_): _This is the current state of the graph_
            edge_index (_list_): _This is a list of edge connections in the graph_
            move_num (_int_): _This is the current time-step_

        Returns:
            _type_: _description_
        """
        if move_num%(self.max_moves-1)==0:
            self.episodes+=1
            self.loss_data=[]
        # Ensure the model is in training mode
        self.train() 
        
        prediction = self.forward(x, edge_index,move_num)
        target = x.detach() 
    
        target = target[:,0].reshape(self.num_nodes,1)
     
        loss = self.loss_f(prediction, target)
        self.loss_data.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()