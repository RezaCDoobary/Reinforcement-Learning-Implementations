import torch
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class QModelNet(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layers  = [64,64], drop_p = 0.3, dueling = False):
        super(QModelNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)

        self.with_dueling = dueling

        self.state_value = nn.Linear(hidden_layers[-1], 1)
        
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, input):
        for linear in self.hidden_layers:
            input = F.relu(linear(input))
            input = self.dropout(input)
        
        if self.with_dueling:
            advantage_function = self.output(input) 
            output =  self.state_value(input) + (advantage_function - torch.mean(advantage_function))
        else:
            output = self.output(input)
        return output