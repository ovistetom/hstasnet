# HSTASNET: Hybrid Spectrogram Time-domain Audio Separation Network

A PyTorch implementation of "Real-time Low-latency Music Source Separation using Hybrid Spectrogram-TasNet", published in ICASSP2024, by S. Venkatesh, A. Benilov, P. Coleman and F. Roskam.

Made for practice, currently incomplete.

Create a dedicated conda environment then install the dependencies:  
`pip3 install -r requirements.txt`

Some remarks:
- The spectrogram branch implemented here is *magnitude-only*, using the phase of the mixture to reconstruct complex STFTs. No particular reason behind this; might be better to stack real and imaginary parts.
- Some routines to process the MUSDB48 and MUSDB18HQ databases are implemented here; might be better to use the tools available in `torchaudio.datasets`.