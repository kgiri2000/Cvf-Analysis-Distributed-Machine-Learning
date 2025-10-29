import torch
import torch.nn as nn

class FeedForwardModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeedForwardModel, self).__init__()
        #Define the layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,output_dim)
        )


    def forward(self, x):
        #Define the forward pass
        return self.network(x)

def build_feed_forward_model_torch(input_dim, output_dim, learning_rate=0.001):
    #Build model and optimizer
    model = FeedForwardModel(input_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()
    return model, optimizer, criterion

