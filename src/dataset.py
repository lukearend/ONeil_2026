import json
import numpy as np
import pandas as pd
import scipy

"""
Notes:
- AP2 in animals 5, 8, and 9 recorded some cells in CA3
- AP1 recorded dentate spikes identified as type 1 (L) or 2 (M)
- for sessions 2020-03-13_14-27-47 through 2020-03-13_19-11-00,
  dentate spikes were only computed in superior, not inferior blade
- 2020-03-11_18-11-25 AP1 matlab LF has incompatible data, time index (size 1220608, 2339881)
- dentate spike detection failed for animal 5
- dentate CSD only shows upper blade for animal 9
"""

SUBJECTS = [5, 8, 9]

SUBJECT_PROBE_DATASET = {
    (5, 'AP1'): 4,
    (5, 'AP2'): 5,
    (8, 'AP1'): 6,
    (8, 'AP2'): 7,
    (9, 'AP1'): 8,
    (9, 'AP2'): 9,
}

SESSION_SUBJECT_RECORDING = {
    '2020-01-16_17-35-48': (5, 8),
    '2020-01-16_17-56-05': (5, 9),
    '2020-01-16_20-25-54': (5, 13),
    '2020-03-11_14-53-27': (8, 6),
    '2020-03-11_15-17-55': (8, 7),
    '2020-03-11_16-14-50': (8, 9),
    '2020-03-11_16-43-44': (8, 10),
    '2020-03-11_17-20-27': (8, 11),
    '2020-03-11_17-44-00': (8, 12),
    '2020-03-13_14-00-04': (9, 1),
    '2020-03-13_14-27-47': (9, 2),
    '2020-03-13_14-51-33': (9, 3),
    '2020-03-13_15-53-43': (9, 4),
    '2020-03-13_16-13-35': (9, 5),
    '2020-03-13_16-35-07': (9, 6),
    '2020-03-13_16-54-24': (9, 7),
    '2020-03-13_17-14-09': (9, 8),
    '2020-03-13_17-34-13': (9, 9),
    '2020-03-13_17-57-33': (9, 10),
    '2020-03-13_18-19-05': (9, 11),
    '2020-03-13_18-49-27': (9, 12),
    '2020-03-13_19-11-00': (9, 13),
}

def load_session_data(session_id):
    subject_id, recording_id = SESSION_SUBJECT_RECORDING[session_id]
    ca3_dataset = SUBJECT_PROBE_DATASET[(subject_id, 'AP2')]

    dg_lf_data_path = f'data/NEUROPIXELS_{subject_id}/NEUROPIXELS/{session_id}/experiment1/recording1/continuous/Neuropix-PXI-100.1/continuous.dat'
    dg_lf_timestamp_path = f'data/NEUROPIXELS_{subject_id}/NEUROPIXELS/{session_id}/experiment1/recording1/continuous/Neuropix-PXI-100.1/timestamps.npy'
    ca_lf_data_path = f'data/NEUROPIXELS_{subject_id}/NEUROPIXELS/{session_id}/experiment1/recording1/continuous/Neuropix-PXI-100.3/continuous.dat'
    ca_lf_timestamp_path = f'data/NEUROPIXELS_{subject_id}/NEUROPIXELS/{session_id}/experiment1/recording1/continuous/Neuropix-PXI-100.3/timestamps.npy'
    dg_csd_data_path = f'data/csd/NEUROPIXELS_{subject_id}_{session_id}_LF1_csd.npy'
    ca_csd_data_path = f'data/csd/NEUROPIXELS_{subject_id}_{session_id}_LF2_csd.npy'
    dg_wav_data_path = f'data/wav_7hz/NEUROPIXELS_{subject_id}_{session_id}_LF1_wav_7hz.npy'
    ca_wav_data_path = f'data/wav_7hz/NEUROPIXELS_{subject_id}_{session_id}_LF2_wav_7hz.npy'
    ca_cluster_data_path = f'data/NEUROPIXELS_{subject_id}/NEUROPIXELS/AP2/{session_id}/cluster_info.tsv'
    ca_ap_cluster_path = f'data/NEUROPIXELS_{subject_id}/NEUROPIXELS/AP2/{session_id}/spike_clusters.npy'
    ca_ap_timestamp_path = f'data/NEUROPIXELS_{subject_id}/NEUROPIXELS/AP2/{session_id}/spike_times.npy'
    ds_data_path = f'data/NEUROPIXELS_{subject_id}/DS/DS_TYPE12/{session_id}.mat'
    cell_info_path = f'ref/features_ca1_ca3dg_good.csv'
    dg_marker_path = f'ref/dg_marker_channel.json'

    dg_lf_timestamps = np.load(dg_lf_timestamp_path, mmap_mode='r')
    ca_lf_timestamps = np.load(ca_lf_timestamp_path, mmap_mode='r')
    dg_data_shape = (len(dg_lf_timestamps), 384)
    ca_data_shape = (len(ca_lf_timestamps), 384)
    dg_lf_memmap = np.memmap(dg_lf_data_path, shape=dg_data_shape, order='C', dtype='int16', mode='r')
    ca_lf_memmap = np.memmap(ca_lf_data_path, shape=ca_data_shape, order='C', dtype='int16', mode='r')
    dg_csd_memmap = np.load(dg_csd_data_path, mmap_mode='r')
    ca_csd_memmap = np.load(ca_csd_data_path, mmap_mode='r')
    dg_wav_memmap = np.load(dg_wav_data_path, mmap_mode='r')
    ca_wav_memmap = np.load(ca_wav_data_path, mmap_mode='r')
    ca_cells = pd.read_csv(ca_cluster_data_path, delimiter='\t')
    ca_ap_clusters = np.load(ca_ap_cluster_path, mmap_mode='r')
    ca_ap_timestamps = np.load(ca_ap_timestamp_path, mmap_mode='r').flatten() // 12 # AP to LF index

    file = scipy.io.loadmat(ds_data_path)
    ds1sup = file['kType1sup'].flatten().astype('bool')
    ds2sup = file['kType2sup'].flatten().astype('bool')
    sample = file['samplesDS'].flatten()
    ds1_sample = sample[ds1sup] - 1 # matlab to python index
    ds2_sample = sample[ds2sup] - 1

    cell_info = pd.read_csv(cell_info_path)
    cell_info = cell_info.loc[cell_info['dataset'] == ca3_dataset].drop(columns='dataset')
    cell_info = cell_info.loc[cell_info['recording'] == recording_id].drop(columns='recording')
    cell_info = cell_info.loc[cell_info['location'] == 'CA3'].drop(columns='location')

    ca_cells = ca_cells[ca_cells['id'].isin(cell_info['id'])]
    spike_mask = np.isin(ca_ap_clusters, ca_cells['id'])
    ca_ap_clusters = ca_ap_clusters[spike_mask]
    ca_ap_timestamps = ca_ap_timestamps[spike_mask]

    with open(dg_marker_path, 'r') as f:
        dg_marker = json.load(f)[session_id]
    
    return {
        'cells': cell_info,
        'dg_marker': dg_marker,
        'dg_timestamp': dg_lf_timestamps,
        'dg_lfp': dg_lf_memmap,
        'dg_csd': dg_csd_memmap,
        'dg_wav': dg_wav_memmap,
        'ca_timestamp': ca_lf_timestamps,
        'ca_lfp': ca_lf_memmap,
        'ca_csd': ca_csd_memmap,
        'ca_wav': ca_wav_memmap,
        'clusters': ca_cells,
        'spike_cluster': ca_ap_clusters,
        'spike_sample': ca_ap_timestamps,
        'ds1_sample': ds1_sample,
        'ds2_sample': ds2_sample,
    }

# These are approximate. They do not have to be as exact as for the Park dataset registration,
# because channels outside HPC are in principle low-theta power and therefore will have small
# contribution to the global phase average. These judgments err on the side of inclusivity to
# avoid leaving out true HPC channels. Made by inspection of CSD covariance matrices in 20251208_decide_LF1_hpc_chan_range.
SESSION_HPC_LIMS = {
    '2020-01-16_17-35-48': (65, 112),
    '2020-01-16_17-56-05': (65, 112),
    '2020-01-16_20-25-54': (69, 116),
    '2020-03-11_14-53-27': (38, 100),
    '2020-03-11_15-17-55': (38, 100),
    '2020-03-11_16-14-50': (38, 100),
    '2020-03-11_16-43-44': (38, 100),
    '2020-03-11_17-20-27': (41, 110),
    '2020-03-11_17-44-00': (41, 110),
    '2020-03-13_14-00-04': (66, 116),
    '2020-03-13_14-27-47': (54, 107),
    '2020-03-13_14-51-33': (54, 107),
    '2020-03-13_15-53-43': (54, 107),
    '2020-03-13_16-13-35': (54, 107),
    '2020-03-13_16-35-07': (54, 107),
    '2020-03-13_16-54-24': (55, 107),
    '2020-03-13_17-14-09': (55, 107),
    '2020-03-13_17-34-13': (55, 107),
    '2020-03-13_17-57-33': (55, 102),
    '2020-03-13_18-19-05': (74, 120),
    '2020-03-13_18-49-27': (49, 100),
    '2020-03-13_19-11-00': (64, 115),
}
