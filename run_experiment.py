from dataclasses import dataclass
from dinr import DisplacementINR

import sys
import os
from os.path import dirname, join, abspath

@dataclass(frozen=True)
class Config:

    # Experiment setup
    do_log: bool = True
    experiment_name: str = 'base'
    project_name: str = 'DINR-STACOM'
    device: str = 'cuda'

    # Data

    training_set: str = '' 
    validation_set: str = '' 
    test_set: str = 'example' 

    # Training Parameters

    epochs: int = 1000
    patience: int = 15
    lr: float = 1e-4
    jacobian_weight: float = 0.001
    prior_weight: float = 1e-4
    batch_size: int = 4
    omega: float = 15

    training_size: int | None = None
    validation_size: int | None = None
    test_size: int | None = None

    cropped_shape: tuple[int, int] = (128, 128)  

    # INR parameters
    in_features: int = 3   
    out_features: int = 2 
    hidden_layers: int = 3
    hidden_dim: int = 256

    #Encoder parameters
    latent_dim: int = 32
    
    # paths

    model_dir: str = './models/'
    results_dir: str = './results/predictions'
    dataset_dir: str = './data/'

def main():
    config = Config()
    dinr = DisplacementINR(config=config)
    dinr.train()
    dinr.test()




if __name__=='__main__':
    sys.path.append(dirname(dirname(abspath(__file__))))
    main()


