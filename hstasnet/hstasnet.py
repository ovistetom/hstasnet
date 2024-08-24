import torch
import torch.nn as nn
import torch.nn.functional as ff

from spec_codec import SpecEncoder, SpecDecoder
from time_codec import TimeEncoder, TimeDecoder
from memory import Memory


class HSTasNet(nn.Module):

    def __init__(self,
                 num_sources,
                 num_channels,
                 time_win_size: int = 1024,
                 time_hop_size: int = 512,
                 time_ftr_size: int = 1500,
                 spec_win_size: int = 1024,
                 spec_hop_size: int = 512,
                 spec_fft_size: int = 1024,
                 rnn_hidden_size: int = 1000,
                 rnn_num_layers: int = 1,
                 ):

        super().__init__()

        self.num_sources = num_sources
        self.num_channels = num_channels
        self.time_win_size = time_win_size
        self.time_hop_size = time_hop_size
        self.time_ftr_size = time_ftr_size
        self.spec_win_size = spec_win_size
        self.spec_hop_size = spec_hop_size
        self.spec_fft_size = spec_fft_size
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_num_layers = rnn_num_layers

        time_feature_size = time_ftr_size
        spec_feature_size = (spec_win_size//2 + 1)
        self.time_feature_size = time_feature_size
        self.spec_feature_size = spec_feature_size

        assert not (time_win_size % time_hop_size), f"time_win_size ({time_win_size}) must be a multiple of time_hop_size ({time_hop_size})"
        assert not (spec_win_size % spec_hop_size), f"spec_win_size ({spec_win_size}) must be a multiple of spec_hop_size ({spec_hop_size})"
        assert time_win_size == spec_win_size, f"time_win_size ({time_win_size}) must be equal to spec_win_size ({spec_win_size})"
        assert time_hop_size == spec_hop_size, f"time_hop_size ({time_hop_size}) must be equal to spec_hop_size ({spec_hop_size})" 

        self.time_encoder = TimeEncoder(
            N=time_win_size,
            O=time_hop_size,
            M=time_ftr_size,
            )

        self.time_decoder = TimeDecoder(
            N=time_win_size,
            O=time_hop_size,
            M=time_ftr_size,
            )

        self.time_rnn_in = Memory(
            input_size=num_channels*time_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            )

        self.time_skip_fc = nn.Linear(
            in_features=num_channels*time_feature_size,
            out_features=rnn_hidden_size,
            )

        self.time_rnn_out = Memory(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            )        

        self.time_mask_fc = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=num_channels*num_sources*time_feature_size,
            )

        self.spec_encoder = SpecEncoder(
            n_win=spec_win_size,
            n_hop=spec_hop_size,
            n_fft=spec_fft_size,
            window='hamming',
            )
        
        self.spec_decoder = SpecDecoder(
            n_win=spec_win_size,
            n_hop=spec_hop_size,
            n_fft=spec_fft_size,
            window='hamming',
            )

        self.spec_rnn_in = Memory(
            input_size=num_channels*spec_feature_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            )
        
        self.spec_skip_fc = nn.Linear(
            in_features=num_channels*spec_feature_size,
            out_features=rnn_hidden_size,
            )        
        
        self.spec_rnn_out = Memory(
            input_size=rnn_hidden_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            )        
        
        self.spec_mask_fc = nn.Linear(
            in_features=rnn_hidden_size,
            out_features=num_channels*num_sources*spec_feature_size,
            )    

        self.hybrid_rnn = Memory(
            input_size=2*rnn_hidden_size,
            hidden_size=2*rnn_hidden_size,
            num_layers=rnn_num_layers,
            )

    def forward(self, waveform, length = None):
        """
        Forward pass through the model.
        
        Args:
            waveform (torch.Tensor): [B, C, L]
            waveform_length (int, optional): The original length of the waveform.

        Returns:
            out (torch.Tensor): [B, S, C, L]
        """

        # Concatenate channels.
        B, C, L = waveform.size()
        x = waveform.view(B*C, L)                                       # (B*C) x L

        # Time domain encoding.
        x_time, x_norm = self.time_encoder(x)                           # (B*C) x T x M

        # Time domain RNN.
        BC, T, M = x_time.size()
        x_time = x_time.view(B, C, T, M)                                # B x C x T x M
        s_time = x_time.permute(0, 2, 1, 3)                             # B x T x C x M
        s_time = s_time.reshape(B, T, C*M)                              # B x T x (C*M)
        y_time = self.time_rnn_in(s_time)                               # B x T x H

        # specuency domain encoding.
        x_spec, x_angl = self.spec_encoder(x)                           # (B*C) x T x F

        # specuency domain RNN.
        BC, T, F = x_spec.size()
        x_spec = x_spec.view(B, C, T, F)                                # B x C x T x F
        s_spec = x_spec.permute(0, 2, 1, 3)                             # B x T x C x F
        s_spec = s_spec.reshape(B, T, C*F)                              # B x T x (C*F)        
        y_spec = self.spec_rnn_in(s_spec)                               # B x T x H

        # Concat time and frequency domain outputs.
        y = torch.cat((y_time, y_spec), dim=2)                          # B x T x (2*H)

        # Hybrid RNN.
        y = self.hybrid_rnn(y)                                          # B x T x (2*H)

        # Split into time and frequency domain.
        H = self.rnn_hidden_size
        y_time, y_spec = torch.split(y, H, dim=2)                       # B x T x H

        # Time-domain RNN and skip-connection
        y_time = self.time_rnn_out(y_time)                              # B x T x H
        s_time = self.time_skip_fc(s_time)                              # B x T x H
        y_time = y_time + s_time                                        # B x T x H

        # Freq-domain RNN and skip-connection
        y_spec = self.spec_rnn_out(y_spec)                              # B x T x H
        s_spec = self.spec_skip_fc(s_spec)                              # B x T x H
        y_spec = y_spec + s_spec                                        # B x T x H

        # Time domain mask-estimation.
        m_time = self.time_mask_fc(y_time)                              # B x T x (S*C*M)
        S, C, M = self.num_sources, self.num_channels, self.time_feature_size
        m_time = m_time.view(B, T, S, C, M)                             # B x T x S x C x H
        m_time = m_time.permute(0, 2, 3, 1, 4)                          # B x C x S x T x H
        x_time = x_time.view(B, 1, C, T, M)                             # B x 1 x C x T x M
        x_time = x_time.expand(B, S, C, T, M)                           # B x S x C x T x M
        y_time = m_time * x_time                                        # B x S x C x T x M

        # Frequency domain mask-estimation.
        m_spec = self.spec_mask_fc(y_spec)                              # B x T x (S*C*F)
        S, C, F = self.num_sources, self.num_channels, self.spec_feature_size
        m_spec = m_spec.view(B, T, S, C, F)                             # B x T x S x C x F
        m_spec = m_spec.permute(0, 2, 3, 1, 4)                          # B x S x C x T x F 
        x_spec = x_spec.view(B, 1, C, T, F)                             # B x 1 x C x T x F
        x_spec = x_spec.expand(B, S, C, T, F)                           # B x S x C x T x F
        y_spec = m_spec * x_spec                                        # B x S x C x T x F

        # Time domain decoding.
        y_time = y_time.reshape(B*S*C, T, M)                            # (B*S*C) x T x M
        x_norm = x_norm.view(B, 1, C, T, 1)                             # B x 1 x C x T x 1
        x_norm = x_norm.expand(B, S, C, T, 1)                           # B x S x C x T x 1
        x_norm = x_norm.reshape(B*S*C, T, 1)                            # (B*S*C) x T x 1
        z_time = self.time_decoder(y_time, x_norm)                      # (B*S*C) x L
        z_time = z_time.view(B, S, C, -1)                               # B x S x C x L

        # Frequency domain decoding.
        y_spec = y_spec.reshape(B*S*C, T, F)                            # (B*S*C) x T x F
        x_angl = x_angl.view(B, 1, C, T, F)                             # B x 1 x C x T x F
        x_angl = x_angl.expand(B, S, C, T, F)                           # B x S x C x T x F
        x_angl = x_angl.reshape(B*S*C, T, F)                            # (B*S*C) x T x F      
        z_spec = self.spec_decoder(y_spec, x_angl)                      # (B*S*C) x L
        z_spec = z_spec.view(B, S, C, -1)                               # B x S x C x L    

        # Sum the outputs.
        out = z_time + z_spec                                           # B x S x C x L

        # Pad the output if necessary.
        if length:
            L = out.size(-1)
            out = ff.pad(out, (0, length-L), 'constant')                # B x S x C x L_padded             

        return out
    
    def _init_args_kwargs(self):

        args = [
            self.num_sources,
            self.num_channels,
            ]

        kwargs = {
            'time_win_size': self.time_win_size,
            'time_hop_size': self.time_hop_size,
            'time_ftr_size': self.time_ftr_size,
            'spec_win_size': self.spec_win_size,
            'spec_hop_size': self.spec_hop_size,
            'spec_fft_size': self.spec_fft_size,
            'rnn_hidden_size': self.rnn_hidden_size,
            }
        
        return args, kwargs
    
    def serialize(self):
        """Serialize the model into a dictionary.
        
        Args:
            model (nn.Module): The model to serialize.
            
        Returns:
            package (dict): A dictionary containing the model's class, arguments, keyword arguments, and state.
        """

        klass = self.__class__
        args, kwargs = self._init_args_kwargs()
        state_dict = self.state_dict()

        package = {
            'klass': klass,
            'args': args,
            'kwargs': kwargs,
            'state_dict': state_dict,
            }
        
        return package        
    
    def save_to_path(self, path):
        """Save the model to a file path.

        Args:
            path (str): The file path to save the model to.
        """
        package = self.serialize()
        torch.save(package, path)

if __name__ == '__main__':

    B, C, L, S = 10, 2, 100000, 4
    x = torch.randn(B, C, L)
    print(f'{x.size() = }')

    model = HSTasNet(
        num_sources=S,
        num_channels=C,
        time_win_size=1024,
        time_hop_size=512,
        time_ftr_size=200,
        spec_win_size=1024,
        spec_hop_size=512,
        spec_fft_size=1024,
        rnn_hidden_size=500,
        rnn_num_layers=1,
        )

    y = model(x, length=L)
    print(f'{y.size() = }')
    