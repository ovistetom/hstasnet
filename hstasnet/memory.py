import torch
import torch.nn as nn
import torch.nn.functional as ff


class Memory(nn.Module):
    """
    A class implementing a memory RNN.

    The model consists of two GRUs with an identity skip connection added to the
    output of the second GRU.
    """

    def __init__(self, 
                 input_size, 
                 hidden_size, 
                 num_layers=1,
                 dropout=0.0,
                 ):
        """
        Args:
            input_size (int):   The size of the input.
            hidden_size (int):  The size of the hidden state.
            num_layers (int, optional): The number of layers. Defaults to 1.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,            
            )

        self.rnn2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            )
        

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor):   [B, H_in]

        Returns:
            z (torch.Tensor):   [B, H_out]
        """

        # RNNs.
        y, _ = self.rnn1(x)                # B x H_out
        z, _ = self.rnn2(y)                # B x H_out

        # Skip-connection.
        z = z + y                           # B x H_out

        return z


if __name__ == '__main__':

    B, H_in, H_out = 80, 1500, 1000
    x = torch.randn(B, 4, H_in)
    print(f'{x.size() = }')

    memory = Memory(
        input_size = H_in,
        hidden_size = H_out,
    )

    y = memory(x)
    print(f'{y.size() = }')
