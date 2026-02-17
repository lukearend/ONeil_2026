#!/usr/bin/env bash

# luke arend
# usage:
# export PASSFILE=/path/to/ssh/pass
# ./etl_dvorak

user="lukea"
host="monk.cns.nyu.edu"
src="/f/fentonlab/data/dinod/ANALYSIS/M32/NEUROPIXELS"
dst="/mnt/home/larend/ceph/data/dvorak"
origin="${user}@${host}:${src}"

# recursive, preserve timestamps, progress aware
# rsync over ssh authenticated by credentials file
copy="sshpass -f ${PASSFILE} rsync -zt --progress"

# comment/uncomment as needed
ANIMAL="NEUROPIXELS_9"
SESSIONS="
2020-03-13_14-00-04
2020-03-13_14-27-47
2020-03-13_14-51-33
2020-03-13_15-53-43
2020-03-13_16-13-35
2020-03-13_16-35-07
2020-03-13_16-54-24
2020-03-13_17-14-09
2020-03-13_17-34-13
2020-03-13_17-57-33
2020-03-13_18-19-05
2020-03-13_18-49-27
2020-03-13_19-11-00
"

# ANIMAL="NEUROPIXELS_8"
# SESSIONS="
# 2020-03-11_14-53-27
# 2020-03-11_15-17-55
# 2020-03-11_16-14-50
# 2020-03-11_16-43-44
# 2020-03-11_17-20-27
# 2020-03-11_17-44-00
# 2020-03-11_18-11-25
# "

# ANIMAL="NEUROPIXELS_5"
# SESSIONS="
# 2020-01-16_17-35-48
# 2020-01-16_17-56-05
# 2020-01-16_20-25-54
# "

for sess in ${SESSIONS} ;
do
    mkdir -p ${dst}/${sess}

    # DAQ start times file
    ${copy} \
    "${origin}/${ANIMAL}/NEUROPIXELS/${sess}*/experiment1/recording1/sync_messages.txt" \
    "${dst}/${sess}/${sess}_sync_messages.txt" ;

    # Animal 5 DS file names differ slightly from corresponding LF/AP files
    # Comment out copy operation in loop and copy them manually:
    # scp ${origin}/NEUROPIXELS_5/DS/DS_TYPE12/2020-01-16_17-35-46_L700.mat \
    #     ${dst}/2020-01-16_17-35-48/2020-01-16_17-35-48_DS_TYPE12.mat
    # scp ${origin}/NEUROPIXELS_5/DS/DS_TYPE12/2020-01-16_17-55-45_L900.mat \
    #     ${dst}/2020-01-16_17-56-05/2020-01-16_17-56-05_DS_TYPE12.mat
    # scp ${origin}/NEUROPIXELS_5/DS/DS_TYPE12/2020-01-16_20-25-52_L400x2.mat \
    #     ${dst}/2020-01-16_20-25-54/2020-01-16_20-25-54_DS_TYPE12.mat
    
    # Dino's dentate spike file
    ${copy} \
    "${origin}/${ANIMAL}/DS/DS_TYPE12/${sess}*.mat" \
    "${dst}/${sess}/${sess}_DS_TYPE12.mat" ;

    for probe in 1 2 ;
    do
        # Dino's kilosort output files
        for file in channel_positions.npy cluster_info.tsv spike_clusters.npy spike_times.npy ;
        do
            ${copy} \
            "${origin}/${ANIMAL}/NEUROPIXELS/AP${probe}/${sess}*/${file}" \
            "${dst}/${sess}/${sess}_AP${probe}_${file}" ;
        done

        # Dino's LFP .mat files
        ${copy} \
        "${origin}/${ANIMAL}/NEUROPIXELS/LFP_MAT${probe}/${sess}*.mat" \
        "${dst}/${sess}/${sess}_LFP_MAT${probe}.mat" ;
    done ;
done
