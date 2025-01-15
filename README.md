# HSTASNET: Hybrid Spectrogram Time-domain Audio Separation Network

A PyTorch implementation of "Real-time Low-latency Music Source Separation using Hybrid Spectrogram-TasNet", published in ICASSP2024, by S. Venkatesh, A. Benilov, P. Coleman and F. Roskam (<https://doi.org/10.48550/arXiv.2402.17701>).

Mainly made for personal practice, some things might not be optimally implemented.

Create a dedicated conda environment then install the dependencies:  
`pip3 install -r requirements.txt`

Some remarks:
- The spectrogram branch implemented here is *magnitude-only*, using the phase of the mixture to reconstruct complex STFTs. No particular reason behind this choice; might be better to stack real and imaginary parts.
- The model is trained using the open MUSDB18-HQ dataset (<https://sigsep.github.io/datasets/musdb.html>). 

Use `torchaudio.datasets.MUSDB_HQ` to download the database, then run the script `utils/preprocess_musdb.py` to run preprocessing routines.