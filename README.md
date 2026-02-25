# ONeil_2026
Code for generating figures in O'Neil et al. 2026 paper ("Temporal coding rather than circuit wiring...").

Uses Python 3 and depends on the packages listed in `requirements.txt`.

# About the raw data
- Used to perform the in-vivo data analyses in the paper (last main text figure).
- The in-vivo data were collected by Dino Dvorak in Fenton lab and used in Dvorak et al. 2021 ("Dentate spikes and external control...").
- Dino used Neuropixels 1.0 to record from CA1-DG (probe 1: medial) and either CA1-DG or CA1-CA3 (probe 2: lateral) in 7 animals.
- Used Kilosort to obtain spike-sorted clusters and spike times from the AP signal recorded at 30 kHz.
- Used k-means on features of the template waveform and spike train to classify cells.
- Used an automatic detection procedure to identify and classify type I (LEC-originating) and type II (MEC-originating) dentate spikes.
- Across all recordings, 257 cells from 3 animals (22 sessions) were localized to CA3; these cells/recordings were subsetted by Luke for analysis in the O'Neil 2026 paper.
- Original raw data dump of Dino's data available at:
  - `<fenton lab data server>:/f/fentonlab/data2/dinod/M32/NEUROPIXELS`
  - `<NYU datalake>:/fentonlab-raw-data/dvorak-ds`

# About the processed data
- The CA3-containing subset of Dino's data were processed by Luke.
- Processed version of Dino's data created by Luke available at:
  - `<NYU datalake>:/fentonlab-data/dvorak-ca3`
- Data products:
  - `{session}_AP{1,2}_channel_positions.npy`
    - position of each channel along the electrode
  - `{session}_AP{1,2}_cluster_info.tsv`
    - information about each spiking cluster detected
  - `{session}_AP{1,2}_spike_clusters.npy`
    - identifier of cluster that generated each spike
  - `{session}_AP{1,2}_spike_times.npy`
    - action potential detection timestamps
    - given as 0-based index from start of AP .dat file
    - sampled at 30 kHz synced to same clock as 2500 Hz LF .dat signal
  - `{session}_DS_TYPE12.mat`
    - dentate spike detection timestamps
    - given as 1-based index from start of LFP_MAT file
  - `{session}_LFP_MAT{1,2}.mat`
    - .mat version of raw LF .dat signal sampled at 2500 Hz
- Documentation:
  - `dvorak_documentation.pdf`
    - original documentation written by Dino
  - `features_ca1_ca3dg_good.xlsx`
    - cell position and type information
  - `features_ca1_ca3dg_good.csv`
    - CSV copy of Excel spreadsheet
  - `dvorak_copy.sh`
    - script used by Luke to extract data for CA3 analysis
  - `src/dataset.py`
    - metadata specific to this dataset used in code

# Data analysis pipeline
- `20260127_fix_2020-01-16_DS_TYPE12.ipynb`
  - fix timestamp issue with one session where Dino's DS timestamps were from Neuronexus instead of Neuropixels recording system
- `20250601_ds_combo_psths.ipynb`
  - load AP, load DS times, create PSTH for all CA3 cells against all DS for all sessions
    - for each DS, note the type of combination based on previous DS type:
    - combo type: ds1/ds2 = 0: I/I, 1: I/II, 2: II/I, 3: II/II
    - write to file `ds_aligned_spikes.pkl`
- `20250602_make_ds_aligned_df.ipynb`
  - re-package results from nested dictionary into dataframe
- `20260130_make_figures.ipynb`
  - generate graphics used in final figures included in the paper

