from json.tool import main
import torch
from torch import nn as nn
import torch.nn.functional as F
import glob
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


## SNR(Signal to Noise Ratio)
## SNR = 20 log10 (A_sig/A_noise)
def get_noisy_sig(sig, noise, snr = 0):
    noise_len = len(noise)
    sig_len = len(sig)

    ## get noise signal
    start_point  = np.random.randint(0, noise_len - sig_len, size = 1)
    _noise = noise[start_point.item(): (start_point + sig_len).item()]
    
    ## sig/noise amplitude := energy
    sig_E = np.mean(sig**2)
    noise_E = np.mean(_noise**2)
    adj_noise_E = sig_E / (10**(snr/20))
    
    ## adjust noise amplitude
    adj_noise = _noise * (adj_noise_E / noise_E) 
    mixed_signal = adj_noise + sig
    return mixed_signal
    

if __name__ == '__main__':
    import os
    print(os.getcwd())
    from dataloader import YesNoDataset
    ## yesno dataset directory
    yesno = YesNoDataset('./waves_yesno/')
    print(len(yesno))
    ## noise (.wav) signal directory
    noise_dir = './noisesB/'
    noise_files = glob.glob(noise_dir+"*.wav")

    selected_noises = [f for f in noise_files if 'const' in f]

    for f in selected_noises:
        noise, sr = librosa.load(f)
        print(noise.shape[0]/sr)
    
    ## signal, noise     
    sig, sr, labels =  yesno[2]
    noise, sr = librosa.load(selected_noises[1])
    mixed = get_noisy_sig(sig, noise)
    sf.write('./sample/mixed_example.wav',data=mixed, samplerate=sr)

        
    ## Plotting mixed Signals
    mixed = get_noisy_sig(sig, noise)
    plt.plot(mixed)
    plt.plot(sig)
    plt.show()



# https://www1.icsi.berkeley.edu/Speech/faq/speechSNR.html