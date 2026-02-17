import numpy as np
import scipy
import pywt # github.com/PyWavelets/pywt
import src.icsd # github.com/espenhgn/iCSD, requires github.com/NeuralEnsemble/python-neo
import quantities as pq # github.com/python-quantities/python-quantities

def dereference(sig, **kwargs):
    return sig - np.mean(sig, keepdims=True, **kwargs)

def lowpass_filter(sig, step_hz=2500, cutoff_hz=1, order=4, **kwargs):
    cutoff = cutoff_hz / (step_hz / 2)
    sos = scipy.signal.butter(btype='lowpass', N=order, Wn=cutoff, output='sos')
    return scipy.signal.sosfiltfilt(sos, sig, **kwargs)

def highpass_filter(sig, step_hz=2500, cutoff_hz=300, order=4, **kwargs):
    cutoff = cutoff_hz / (step_hz / 2)
    sos = scipy.signal.butter(btype='highpass', N=order, Wn=cutoff, output='sos')
    return scipy.signal.sosfiltfilt(sos, sig, **kwargs)

def gaussian_filter(sig, step_um=40, std_um=40, radius_um=None, **kwargs):
    assert std_um % step_um == 0, 'require std to divide evenly by step size'
    std = std_um // step_um
    if radius_um is not None:
        assert radius_um % step_um == 0, 'require radius to divide evenly by step size'
        radius = radius_um // step_um
    else:
        radius = std*3
    return scipy.ndimage.gaussian_filter1d(sig, sigma=std, radius=radius, **kwargs)

def csd_transform(sig, step_um=40, conductivity=0.3):
    x = step_um * 1e-6 * np.arange(sig.shape[1]) * pq.m
    sigma = conductivity * pq.S / pq.m
    lfp = sig.T * pq.V
    csd = icsd.StandardCSD(lfp, coord_electrode=x, sigma=sigma).get_csd()
    return np.array(csd).T

def wavelet_transform(sig, step_hz=2500, freq_hz=7, template='cmor1.5-1.0', **kwargs):
    wav_freq = freq_hz / step_hz # [cycle/sample]
    wav = pywt.ContinuousWavelet(template)
    scale = pywt.frequency2scale(wav, wav_freq)
    coefs, _ = pywt.cwt(sig, scales=scale, wavelet=wav, **kwargs)
    return np.array(coefs[0])

def load_neuropixels(file, sample_start=0, sample_stop=2500, chan_start=0, chan_stop=384):
    assert chan_start % 4 == 0
    assert chan_stop % 4 == 0
    with open(file.replace('.bin', '.meta'), 'r') as f:
        metadata = dict([kv.strip('~').split('=') for kv in f.readlines()])
        n_chan = int(metadata['nSavedChans'])
        n_byte = int(metadata['fileSizeBytes'])
        n_sample = (n_byte // 2) // n_chan
        assert n_sample == (n_byte // 2) // 385
    assert chan_stop <= 384
    assert sample_stop <= n_sample
    view = np.memmap(file, dtype='int16', mode='r', order='C', shape=(n_sample, n_chan))
    view = view[sample_start:sample_stop]
    chan_191 = (view[:, 191-4] + view[:, 191+4]) // 2
    DAQ = np.array(view[:, chan_start:chan_stop])
    if chan_start <= 191 & 191 < chan_stop:
        # replace dead channel 191 with average over D-V neighbors
        DAQ[:, 191-chan_start] = chan_191
    return DAQ
    
def extract_LFP(DAQ, smooth_um=None):
    """ (samples, NP channels) -> (4, samples, 40 µm channels) """
    assert DAQ.shape[1] % 4 == 0
    n_shank, n_sample, n_channel = 4, DAQ.shape[0], DAQ.shape[1] // 4
    DAQ0 = DAQ[:, 0::4] # 0:   0 µm DV,  +0 µm ML, 40 µm step
    DAQ1 = DAQ[:, 1::4] # 1:   0 µm DV, +32 µm ML, 40 µm step
    DAQ2 = DAQ[:, 2::4] # 2: +20 µm DV, +16 µm ML, 40 µm step
    DAQ3 = DAQ[:, 3::4] # 3: +20 µm DV, +48 µm ML, 40 µm step
    DAQ = np.array([DAQ0, DAQ1, DAQ2, DAQ3])
    LFP = np.zeros((n_shank, n_sample, n_channel))
    for i in range(n_shank):
        sig = DAQ[i]
        sig = dereference(sig, axis=1) # use mean across sites as reference voltage
        sig = highpass_filter(sig, axis=0, step_hz=2500, cutoff_hz=1) # remove slow-timescale fluctuation
        sig = lowpass_filter(sig, axis=0, step_hz=2500, cutoff_hz=300) # remove fast-timescale fluctuation
        LFP[i] = sig
    if smooth_um:
        LFP = gaussian_filter(LFP, step_um=40, std_um=smooth_um, axis=-1)
    return LFP

def extract_CSD(LFP, smooth_um=40):
    """ (4, samples, 40 µm channels) -> (samples, 20 µm channels) """
    assert LFP.shape[0] == 4
    n_shank, n_sample, n_channel = LFP.shape
    CSD = np.zeros((n_shank, n_sample, n_channel))
    for i in range(n_shank):
        sig = LFP[i]
        sig = csd_transform(sig, step_um=40) # estimate CSD from LFP
        sig = scipy.stats.zscore(sig, axis=None) # correct for gain difference between shanks
        CSD[i] = sig
    # assume CSD has variance along D-V axis >> variance M-L,
    # this warrants averaging across shanks with same D-V displacement
    CSDA = CSD[0] + CSD[1] # A:   0 µm DV, +16 µm ML, 40 µm step
    CSDB = CSD[2] + CSD[3] # B: +20 µm DV, +32 µm ML, 40 µm step
    # assume CSD has approximate radial symmetry about D-V axis,
    # this warrants interleaving shanks with opposing displacement over M-L
    CSD = np.array([CSDA, CSDB]).transpose(1, 2, 0) # samples, channels, 2 shanks
    CSD = CSD.reshape(n_sample, n_channel * 2) # 0 µm DV, +24 µm ML, 20 µm step
    if smooth_um:
        # recommend 40 µm for HPC, 100 µm for PFC
        CSD = gaussian_filter(CSD, step_um=20, std_um=smooth_um, axis=-1)
    return CSD

def extract_WAV(sig, scales_hz):
    """ (samples, channels) -> (scales, samples, channels) """
    n_scale, n_sample, n_channel = len(scales_hz), sig.shape[0], sig.shape[1]
    WAV = np.zeros((n_scale, n_sample, n_channel), dtype='complex')
    for i, f in enumerate(scales_hz):
        WAV[i] = wavelet_transform(sig, axis=0, step_hz=2500, freq_hz=f)
    return WAV
