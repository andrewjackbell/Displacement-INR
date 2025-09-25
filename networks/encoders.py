#
# Adapted from: https://github.com/uncbiag/NePhi
# 

import torch
import torch.nn as nn
from networks.layers import convBlock, FullyConnectBlock

def compute_encoder_output_shape(stride_list, shape, prod=True):
    from functools import reduce

    out_shape = [
        reduce(lambda x, y: (x - 1) // y + 1, [shape[i]] + stride_list)
        for i in range(len(shape))
    ]
    return out_shape if not prod else reduce(lambda x, y: x * y, out_shape)


class GlobalAndLocalEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        data_shape,
        enc_filters=[16, 32, 64, 128, 256],
        local_feature_layer_num=3,
        kernel_size=3,
        batchnorm=True,
    ) -> None:
        super().__init__()

        dim = len(data_shape)
        self.local_encoder = convBlock(
            enc_filters[local_feature_layer_num],
            latent_dim,
            dim=dim,
            batchnorm=batchnorm,
        )

        feature_net = nn.ModuleList()
        enc_filters = [2] + enc_filters
        for i in range(len(enc_filters) - 1):
            feature_net.append(
                convBlock(
                    enc_filters[i],
                    enc_filters[i + 1],
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    bias=True,
                    dim=dim,
                    batchnorm=batchnorm,
                )
            )
        self.feature_encoder = feature_net

        global_net = nn.ModuleList()
        global_net.append(nn.Flatten())
        dim = compute_encoder_output_shape([2] * (len(enc_filters) - 1), data_shape)
        global_net.append(FullyConnectBlock(enc_filters[-1] * dim, 256))
        global_net.append(FullyConnectBlock(256, latent_dim, nonlinear=None))
        self.global_encoder = nn.Sequential(*global_net)
        self.local_feature_layer_num = (
            local_feature_layer_num
            if local_feature_layer_num >= 0
            else len(enc_filters) - 1 + local_feature_layer_num
        )
        self.latent_dim = latent_dim * 2

    def forward(self, I1, I2):
        x = torch.cat([I1, I2], dim=1)
        local_features = None
        for i in range(len(self.feature_encoder)):
            x = self.feature_encoder[i](x)
            if i == self.local_feature_layer_num:
                local_features = x
        latent_code_global = self.global_encoder(x)  # BxC
        latent_code_local = self.local_encoder(local_features)  # BxCxHxW

        return latent_code_global, latent_code_local

if __name__ == "__main__":
    # Example usage
    encoder = GlobalAndLocalEncoder(latent_dim=64, data_shape=(128, 128))
    I1 = torch.randn(8, 1, 128, 128)  # Batch of one-channel images
    I2 = torch.randn(8, 1, 128, 128)  # Another batch of one-channel images
    global_code, local_code = encoder(I1, I2)
    print("Global latent code shape:", global_code.shape)
    print("Local latent code shape:", local_code.shape)
    
