import torch
import torch.nn as nn
import torch.nn.functional as F

class BackboneToBiFPNAdapter(nn.Module):
    """
    A class to create an adapter layer that connects the output of a backbone network
    to the input of a BiFPN (Bidirectional Feature Pyramid Network) layer.

    Parameters:
    - num_bifpn_features: The number of output features for the BiFPN.
    - backbone_output_channels: A dictionary where keys are layer names and
                                values are the number of output channels from the backbone.
    """

    def __init__(self, num_bifpn_features=256, backbone_output_channels=None):
        super().__init__()

        if backbone_output_channels is None:
            backbone_output_channels = {
                "layer1": 40,
                "layer2": 80,
                "layer3": 160,
                "layer4": 320,
            }

        self.adapters = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=num_bifpn_features,
                      kernel_size=1,
                      stride=1,
                      padding=0)
            for _, in_channels in backbone_output_channels.items()
        ])


    def forward(self, x_in_stages):
        """
        Forward pass through the adapter layers. Assumes x is a dictionary where keys
        match those in backbone_output_channels, and values are the corresponding feature maps.

        Parameters:
        - x_in_stages: A list of tensors from the backbone layers.

        Returns:
        - A list of feature maps processed to match the BiFPN input dimensions.
        """
        return [ adapter(x_in_stage) for adapter, x_in_stage in zip(self.adapters, x_in_stages) ]
