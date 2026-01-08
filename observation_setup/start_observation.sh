# https://askubuntu.com/questions/46627/how-can-i-make-a-script-that-opens-terminal-windows-and-executes-commands-in-the
# https://superuser.com/questions/454907/how-to-execute-a-command-in-screen-and-detach

# Read observation configuration file (.yaml)
source observation_config.sh observation_conf.yaml

# Ramdisk_10GB
sudo umount $SOURCE
rm -rf $SOURCE
mkdir $SOURCE
sudo mount -t tmpfs -o size=10G Ramdisk_10G $SOURCE
sleep 0.5

# Ramdisk_20GB
sudo umount $TEMPSOURCE
rm -rf $TEMPSOURCE
mkdir $TEMPSOURCE
sudo mount -t tmpfs -o size=20G Ramdisk_20G $TEMPSOURCE
sleep 0.5

# Ramdisk_70GB
sudo umount $TRANSIENTBUF
rm -rf $TRANSIENTBUF
mkdir $TRANSIENTBUF
sudo mount -t tmpfs -o size=70G Ramdisk_70G $TRANSIENTBUF
sleep 0.5

# GBD_HDF5ARCHIVE
# sudo blkid
sudo umount $Archive_dir_HDF5
sudo mkdir $Archive_dir_HDF5
drive_block=/dev/sda5
sudo mount $drive_block $Archive_dir_HDF5

# BOOKEEP
rm -rf $BOOKEEP
mkdir $BOOKEEP

# Change Ethernet port to eth0
sudo bash initiate_eth0_for_PDR.sh

#Terminal 1: - To initiate UDP pkt GULP acqusition
echo
echo UDP pkt GULP acqusition
#cd $SCRIPTS_DIR

echo $GULP_script $ACQ_file_prefix $ETHERNET_name $SOURCE

#gnome-terminal --tab --title="GULP" -- bash -c  "sudo $GULP_script $ACQ_file_prefix $ETHERNET_name $SOURCE; exec bash" &&

sudo screen -dmS "GULP" bash -c " cd $SCRIPTS_DIR ; sudo bash $GULP_script $ACQ_file_prefix $ETHERNET_name $SOURCE; exec bash" &&

#Terminal 2: - To RSYNC data to temp from ramdisk
echo
echo RSYNC data from Ramdisk

echo $rsync_script $SOURCE $TEMPSOURCE $TRANSIENTBUF $BOOKEEP $user

#gnome-terminal --tab --title="RSYNC PDR data" -- bash -c  "sudo $rsync_script $SOURCE $TEMPSOURCE $TRANSIENTBUF $BOOKEEP $user; exec bash" &&

sudo screen -dmS "RSYNC_PDR_data" bash -c " cd $SCRIPTS_DIR ; sudo bash $rsync_script $SOURCE $TEMPSOURCE $TRANSIENTBUF $BOOKEEP $user; exec bash" &&

#Terminal 3: - To initiate Multithread data processing
echo 
echo Multithread data processing

echo python3 $Multithread_spawn_script $TEMPSOURCE $Multithread_script $rf_path_phase $Archive_dir_HDF5

#gnome-terminal --tab --title="DATA_reduction" -- bash -c  "python3 $Multithread_spawn_script $TEMPSOURCE $Multithread_script $rf_path_phase $Archive_dir_HDF5; exec bash" 

screen -dmS "DATA_reduction" bash -c " cd $SCRIPTS_DIR ; python3 $Multithread_spawn_script $TEMPSOURCE $Multithread_script $rf_path_phase $Archive_dir_HDF5; exec bash" &&

#Terminal 4: Transient buffer manager
echo
echo Transient buffer manager

echo $Tran_Buf_Mgr_script $TRANSIENTBUF $BOOKEEP $TRIGGER $DESTINATION $Remote_TRANSIENTBUF $TEST_FOR_FILE $NFILES2KEEP_in_BUFF $NUM_FILES_TX

#gnome-terminal --tab --title="Transient_buffer_manager" -- bash -c  "$Tran_Buf_Mgr_script $TRANSIENTBUF $BOOKEEP $TRIGGER $DESTINATION $Remote_TRANSIENTBUF $TEST_FOR_FILE $NFILES2KEEP_in_BUFF $NUM_FILES_TX; exec bash" &&

screen -dmS "Transient_buffer_manager" bash -c " cd $SCRIPTS_DIR ; bash $Tran_Buf_Mgr_script $TRANSIENTBUF $BOOKEEP $TRIGGER $DESTINATION $Remote_TRANSIENTBUF $TEST_FOR_FILE $NFILES2KEEP_in_BUFF $NUM_FILES_TX; exec bash" &&

#Terminal 5: Watch data

#gnome-terminal --tab --title="Watch_data" -- bash -c  "watch -n 1 'echo $SOURCE; ls -ltrh $SOURCE; echo $TEMPSOURCE; ls -ltrh $TEMPSOURCE; echo $TRANSIENTBUF; ls -ltrh $TRANSIENTBUF'; exec bash"

screen -dmS "Watch_data" bash -c " cd $SCRIPTS_DIR ; watch -n 1 'echo $SOURCE; ls -ltrh $SOURCE; echo $TEMPSOURCE; ls -ltrh $TEMPSOURCE; echo $TRANSIENTBUF; ls -ltrh $TRANSIENTBUF'; exec bash" &&

# Command to stop observation
# stop_observation

stop_observation () {
    screen -ls | grep Detached | cut -d. -f1 | awk '{print $1}' | xargs kill
    sudo screen -ls | grep Detached | cut -d. -f1 | awk '{print $1}' | xargs sudo kill
}

