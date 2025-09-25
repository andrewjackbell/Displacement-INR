#
# Adapted from: https://github.com/uncbiag/NePhi
# 

import torch
import torch.nn as nn
import numpy as np

class SinLayer(nn.Module):
    """
    A linear layer followed by a sine activation, as used in SIREN networks.
    """
    def __init__(self, in_features, out_features, omega=1.0, is_first=False):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features)
        
        # Initialize weights
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1 / in_features, 1 / in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / in_features) / omega, 
                                           np.sqrt(6 / in_features) / omega)
    
    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))

class ModulatedSiren(nn.Module):
    """Simplified SIREN network with latent vector modulation"""
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 latent_dim=32, omega=30.0):
        super().__init__()
        
        self.omega = omega
        self.latent_dim = latent_dim
        
        self.net = nn.ModuleList()
       
        self.net.append(SinLayer(in_features, hidden_features, omega, is_first=True))
        
        for _ in range(hidden_layers):
            self.net.append(SinLayer(hidden_features, hidden_features, omega))
        
        self.final_layer = nn.Linear(hidden_features, out_features)
        
        # Modulation layers (one for each hidden layer)
        self.modulators = nn.ModuleList()
        for _ in range(hidden_layers + 1):  # +1 for first layer
            self.modulators.append(nn.Sequential(
                nn.Linear(latent_dim, hidden_features),
                nn.ReLU(),
                nn.Linear(hidden_features, hidden_features)
            ))
    
    def forward(self, coords, latent):
        """
        Args:
            coords: input coordinates (batch_size, in_features)
            latent: conditioning vector (batch_size, latent_dim)
        Returns:
            output: network output (batch_size, out_features)
        """
        x = coords
        
        for i, (layer, mod) in enumerate(zip(self.net, self.modulators)):

            mod_weights = mod(latent)
            

            x = layer(x)
            
            # Modulate output
            x = x * mod_weights
        

        return self.final_layer(x)
