#!/bin/bash

SOURCE=$1 #'/home/gbdobserver/Ramdisk_10GB'
TEMPSOURCE=$2 #'/home/gbdobserver/Ramdisk_20GB'
TRANSIENTBUF=$3 #'/home/gbdobserver/Ramdisk_70GB'
FILELIST=$4/filelist.$(date +%s)    #/tmp/filelist.$(date +%s) #$TARGET/
INTERVAL=0.05

user=$5

cd $SOURCE

x=1

while [ $x -le 5 ]
do
    #cd $SOURCE
    find *.pcap -type f -cmin +$INTERVAL > $FILELIST
    cat $FILELIST
    #sleep 1
    suffix="_$(date +%d_%m_%Y_%H_%M_%s).PCAP_PDR"
    if [ $(cat $FILELIST | wc -l) -ne "0" ]
    then
       rsync -avP --chown=$user:$user --remove-source-files --files-from=$FILELIST $SOURCE/ $TEMPSOURCE/
       while read -r line; do mv "$TEMPSOURCE/$line" "$TEMPSOURCE/$line$suffix"; done < $FILELIST
       # 21/3/24 - copy to the pink block 
       while read -r line; do rsync -avP "$TEMPSOURCE/$line$suffix" "$TRANSIENTBUF/$line$suffix"; done < $FILELIST
       
    fi
    rm -f $FILELIST

done

#-------------------------------------------------------------------------------------------
#cd $TARGET
#while read -r line; do $(/usr/bin/python3 /home/gbdobserver/Documents/PDR_ACQ/obs_scripts/pdr_pcap_file_to_HDF5_allX_25_12_2023.py $TEMPSOURCE/*$suffix) ; done < $FILELIST

 #rm -f $HDF5FILELIST
       
#find *.hdf5 -type f > $HDF5FILELIST
#while read -r line; do mv "$TARGET/$line" "$Data_archive"; done < $HDF5FILELIST
#mv "${SOURCE}/*.hdf5" "${Data_archive}"
#find /tmp/dir/ -cmin +2 | 
#rsync -ogr --chown=user:group --remove-source-files --include-from=- /tmp/dir/ /tmp/dir2/

#--remove-source-files
# sudo /home/summer/Documents/PDR_DATA_POST_PROC/gulp_acquire_19_12_2023.sh drift_scan eno1
#sudo ./rsync_PDR_data_from_Ramdisk_20_12_2023.sh

#sudo --preserve-env=HOME /usr/bin/env ~/Documents/PDR_ACQ/obs_scripts/rsync_PDR_data_from_Ramdisk_with_echo_allX_02_01_2024.sh 
