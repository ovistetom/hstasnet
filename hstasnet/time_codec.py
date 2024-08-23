import torch
import torch.nn as nn
import torch.nn.functional as ff


EPS = 1e-8


class TimeEncoder(nn.Module):

    def __init__(self,
                 N: int = 1024,
                 O: int = 512,
                 M: int = 1500,
                 ):
        """
        Initialize a new TimeEncoder object.

        Args:
            N (int, optional): The number of channels. Defaults to 1024.
            O (int, optional): The number of overlap. Defaults to 512.
            M (int, optional): The number of filters. Defaults to 1500.
        """        

        super().__init__()

        self.N = N
        self.O = O
        self.M = M

        self.conv = nn.Linear(
            in_features=N,
            out_features=M,
            )

        self.gate = nn.Linear(
            in_features=N,
            out_features=M,
            )   
                 
        self.relu = ff.relu
        self.sigmoid = torch.sigmoid

    def forward(self, waveform):
        """
        Forward pass through the model.
        
        Args:
            waveform (torch.Tensor): [B, L]
        
        Returns:
            x_norm (torch.Tensor): [B, T, 1]
            x (torch.Tensor): [B, T, M]
        """

        # Signal to overlapped frames.
        x = waveform.unfold(1, size=self.N, step=self.O)        # B x T x N
        
        # Normalize the frames.
        x_norm = torch.norm(x, p=2, dim=2, keepdim=True)        # B x T x 1
        x = x / (x_norm + EPS)                                  # B x T x N

        # Linear transformation.
        conv = self.relu(self.conv(x))                          # B x T x M
        gate = self.sigmoid(self.gate(x))                       # B x T x M

        # Compute mixture weights.
        x = conv * gate                                         # B x T x M

        return x, x_norm


class TimeDecoder(nn.Module):

    def __init__(self,
                 N: int = 1024,
                 O: int = 512,
                 M: int = 1500,
                 ):
        """
        Initialize a new TimeDecoder object.

        Args:
            N (int, optional): The number of channels. Defaults to 1024.
            O (int, optional): The number of overlap. Defaults to 512.
            M (int, optional): The number of filters. Defaults to 1500.
        """                

        super().__init__()

        self.N = N
        self.O = O
        self.M = M

        self.linear = nn.Linear(M, N)

    def forward(self, waveform_encoding, waveform_norm, waveform_length = None):
        """
        Forward pass through the model.

        Args:
            waveform_encoding (torch.Tensor): [B, T, M]
            waveform_norm (torch.Tensor): [B, T, 1]
        
        Returns:
            x (torch.Tensor): [B, L]
        """

        # Linear transformation (decoder filtering).
        x = self.linear(waveform_encoding)                  # B x T x N

        # Reverse L2 normalization.
        x = x * waveform_norm                               # B x T x N

        # Overlapped frames to signal.
        x = overlap_add(x, self.N//self.O)                  # B x L

        if waveform_length:
            L = x.size(-1)
            x = ff.pad(x, (0, waveform_length-L), 'constant')            # B x L_padded     

        return x
    

def overlap_add(frames, overlap_ratio=2):
    """
    Overlap-adds a batch of frames back into a batch of signals.

    Args:
        frames (torch.Tensor): Batch of frames, size [batch_size, num_frames, frame_size].
        hop_size (int): The hop size (in samples) used when the frames where created.

    Returns:
        signal (torch.Tensor): Batch of signal, size [batch_size, signal_size].
    """

    batch_size, num_frames, frame_size = frames.size()
    overlap_size = frame_size // overlap_ratio

    # Recover the original signal size.
    signal_size = (num_frames - 1) * overlap_size + frame_size

    # Initialize the signal tensor.
    signal = torch.zeros(batch_size, signal_size, dtype=frames.dtype, device=frames.device)

    # Sum the overlapping frames.
    for i in range(num_frames):
        start_idx = i * overlap_size
        end_idx = start_idx + frame_size
        signal[:, start_idx:end_idx] += frames[:, i, :]
    
    # Handle the edge case.
    signal[:, :overlap_size] *= overlap_ratio
    signal[:, -overlap_size:] *= overlap_ratio
    for i in range(1, overlap_ratio-1):
        start_idx = i * overlap_size
        end_idx = start_idx + overlap_size
        signal[:, start_idx:end_idx] *= (overlap_ratio/(i+1))
        signal[:, -end_idx:-start_idx] *= (overlap_ratio/(i+1))
        
    # Normalize the reconstructed signal.
    signal /= overlap_ratio

    return signal


if __name__ == '__main__':

    # Define input.
    B, C, S, L = 10, 2, 4, 500000
    x = torch.randn(B, L)
    print(f'{x.size() = }')

    # Define encoder.
    N, O, M = 1024, 512, 1000
    encoder = TimeEncoder(N=N, O=O, M=M)

    # Compute output.
    y, y_norm = encoder(x)
    print(f'{y.size() = }')
    print(f'{y_norm.size() = }')

    # Define decoder.
    decoder = TimeDecoder(N=N, O=O, M=M)

    # Compute output.
    z = decoder(y, y_norm, waveform_length=None)
    print(f'{z.size() = }')

    # Check overlap-add.
    x = torch.ones(10, 16)
    y = x.unfold(1, size=4, step=2)
    z = overlap_add(y, 2)
    print(f'{x[0] = }')
    print(f'{z[0] = }')
