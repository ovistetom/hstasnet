import torch
import torch.nn as nn
import torch.nn.functional as ff
import torchaudio.transforms as tt


class SpecEncoder(nn.Module):

    def __init__(self,
                 n_win: int = 1024,
                 n_hop: int = 512,
                 n_fft: int = 1024,
                 window: str = 'hamming',
                 ):
        """
        Initialize a new FreqEncoder object.

        Args:
            n_win (int, optional): The window size. Defaults to 1024.
            n_hop (int, optional): The hop size. Defaults to 512.
            n_fft (int, optional): The FFT size. Defaults to 1024.
            window (str, optional): The window type. Defaults to 'hann'.
            device (str, optional): The device to use. Defaults to 'cpu'.
        """

        super().__init__()

        self.n_win = n_win
        self.n_hop = n_hop
        self.n_fft = n_fft
        
        if window == 'bartlett':
            window_fn = torch.bartlett_window
        elif window == 'blackman':
            window_fn = torch.blackman_window
        elif window == 'hamming':
            window_fn = torch.hamming_window
        elif window == 'hann':
            window_fn = torch.hann_window
        elif window == 'kaiser':
            window_fn = torch.kaiser_window
        else:
            raise Exception(f"Invalid window type for STFT : '{window}'.")
        
        self.transform = tt.Spectrogram(
            n_fft=n_fft, 
            hop_length=n_hop, 
            power=None, 
            win_length=n_win,
            window_fn=window_fn,
            normalized=True,
            onesided=True,
            center=False,
            )        

    def forward(self, waveform):
        """
        Forward pass through the model.

        Args:
            waveform (torch.Tensor): [B, L]

        Returns:
            spec_magn (torch.Tensor): [B, T, F]
            spec_angl (torch.Tensor): [B, T, F]
        """
        
        # Compute complex spectrogram.
        spec = self.transform(waveform)             # [B, F, T]

        # Compute magnitude spectrogram.
        spec_magn = torch.abs(spec)                 # [B, F, T]
        spec_magn = spec_magn.permute(0, 2, 1)      # [B, T, F]

        # Compute angle spectrogram.
        spec_angl = torch.angle(spec)
        spec_angl = spec_angl.permute(0, 2, 1)      # [B, T, F]

        return spec_magn, spec_angl

class SpecDecoder(nn.Module):

    def __init__(self,
                 n_win: int = 1024,
                 n_hop: int = 512,
                 n_fft: int = 1024,
                 window: str = 'hamming',
                 ):
        """
        Initialize a new FreqEncoder object.

        Args:
            n_win (int, optional): The window size. Defaults to 1024.
            n_hop (int, optional): The hop size. Defaults to 512.
            n_fft (int, optional): The FFT size. Defaults to 1024.
            window (str, optional): The window type. Defaults to 'hann'.
        """
                
        super().__init__()

        self.n_win = n_win
        self.n_hop = n_hop
        self.n_fft = n_fft
        
        if window == 'bartlett':
            window_fn = torch.bartlett_window
        elif window == 'blackman':
            window_fn = torch.blackman_window
        elif window == 'hamming':
            window_fn = torch.hamming_window
        elif window == 'hann':
            window_fn = torch.hann_window
        elif window == 'kaiser':
            window_fn = torch.kaiser_window
        else:
            raise Exception(f"Invalid window type for STFT : '{window}'.")
                
        self.transform = tt.InverseSpectrogram(
            n_fft=n_fft, 
            hop_length=n_hop, 
            win_length=n_win,
            window_fn=window_fn,
            normalized=True,
            onesided=True,
            center=False,
            )        
        

    def forward(self, spec_magn, spec_angl, waveform_length=None):
        """
        Forward pass through the model.

        Args:
            spec_magn (torch.Tensor): [*, T, F]
            spec_angl (torch.Tensor): [*, T, F]        
            waveform_length (int, optional): The original length of the waveform.

        Returns:
            waveform (torch.Tensor): [*, L]
        """

        spec_real = spec_magn * torch.cos(spec_angl)                # [B, T, F]
        spec_imag = spec_magn * torch.sin(spec_angl)                # [B, T, F]

        spec = torch.complex(spec_real, spec_imag)                  # [B, T, F]
        spec = spec.permute(0, 2, 1)                                # [B, F, T]

        waveform = self.transform(spec, length=waveform_length)     # [B, L]

        return waveform

if __name__ == '__main__':

    # Define input.
    B, C, L = 10, 2, 500000
    x = torch.randn(B, L)
    print(f'{x.size() = }')

    # Define encoder.
    encoder = SpecEncoder(window='hamming')

    # Compute output.
    y_magn, y_angl = encoder(x)
    _, F, T = y_magn.size()
    print(f'{y_magn.size() = }')

    # Define decoder.
    decoder = SpecDecoder(window='hamming')

    # Compute output.
    z = decoder(y_magn, y_angl, waveform_length=None)

    print(f'{z.size() = }')

    # Compare input and output.
    print(f'{x[0, :8] = }')
    print(f'{z[0, :8] = }')
