import torch
from torch.nn import Module
import networkx as nx
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
class uncertainty_estimator(Module) :
    """_This module is the same as the decentralized neural state estimator, except that it
    takes in the observations from every agent, not just one._
    """
    def __init__(self, feature_dim:int, hidden_dim:int, out_dim:int, num_nodes:int, num_agents:int,max_moves:int):
        super().__init__()
        
        if torch.cuda.is_available():
            self.device="cuda"
        else: 
            self.device="cpu"
        #print(f"Self Device is {self.device}")
        self.episodes=0
        self.num_nodes=num_nodes
        self.max_moves=max_moves
        
            # optional second layer for better graph depth                  
        self.lin = torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.GELU(), torch.nn.Linear(3,1))
        self.lin=self.lin.to(self.device)
        #print(self.device)

        self.loss_data=[]
        self.optimizer = torch.optim.Adam(self.parameters(),lr=1e-3)
        self.loss_f = torch.nn.MSELoss()
        self.current_loss=0

    def forward(self,x,edge_index,move_num):
        x =self.lin(x)

        #print(f"shape of x here iss {x.shape}")
        return x
    
    def update_estimator(self, x, last_x, edge_index,move_num):
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
        #print(f" in update estimator, x shape is {x.shape}, last_x shape is {last_x.shape}")
        x=x.to(self.device)
        last_x=last_x.to(self.device)

        
        self.train() 
        prediction = self.forward(x=last_x, edge_index=edge_index,move_num=move_num)
        #print(f"prediction size is {prediction.size}")
        target = x[:,0].reshape(self.num_nodes,1)
        #print(f"target size is {target.size}")

        loss = self.loss_f(prediction, target)
        self.loss_data.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()