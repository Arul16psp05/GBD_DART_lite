
# GBD_DART_lite_obs_script_V1.sh < 119 > < J0953+0755 > < 0 >

Nfiles=$1
PSR=$2
wait_time=$3 #in seconds
CH=0 # full stokes
Freq=175

master_log_dir='/home/dsp/PDR_acquire_setup/master_obs_log'
log_time=$(date '+%d_%m_%Y')
obs_logfile=$master_log_dir"/"$PSR"_"$log_time"_observation.log"
#obs_logfile=$PSR"_"$log_time"_observation.log"
echo_time(){ echo $(date "+%d-%m-%Y %H:%M") "$@" ;}

echo Waiting for $PSR observation...
echo_time "Waiting for $PSR observation..." >> $obs_logfile
sleep $wait_time

echo Observation started
echo_time "Observation started..." >> $obs_logfile 
date 

Transient_BUFF='/data/TRANSIENT_BUFFER'
parent_dir='/data/dsp/GBD_TRANSIENT_BUFFER'
processed_data=$parent_dir/processed_data
slip_check_script='/home/dsp/PDR_acquire_setup/pcap_counter_check/pkt_count_checker_and_filter_12_05_2025.py'
#slip_check_script='/home/dsp/PDR_acquire_setup/pcap_counter_check/pkt_count_checker_and_patcher_05_05_2025.py'
pipeline_script_path='/home/dsp/PDR_acquire_setup/dada_scripts'
#pipeline_script=$pipeline_script_path/PDR2DADA2FITS_HDR_CH_select_multithread_with_FIL_file_RFIclean_02_05_2025.py
#pipeline_script=$pipeline_script_path/PDR2DADA2FITS_HDR_CH_select_multithread_full_Stokes_FITS_12_08_2025.py
pipeline_script=$pipeline_script_path/PDR2DADA2FITS_HDR_CH_select_dada_multithread_full_Stokes_FITS_09_09_2025.py
phase_file=$pipeline_script_path/phase/rf_path_phase.dat

# Clear Transient Buffer

#echo "Clearing Transient Buffer ... rm $Transient_BUFF/* ..."
#echo_time "Clearing Transient Buffer ... rm $Transient_BUFF/* ..." >> $obs_logfile
#rm $Transient_BUFF/*

# --------------- DM from parfile ------------------
pardir='/home/dsp/parfiles'
parfile=$(ls $pardir/$PSR.par)
DM=$(cat $parfile | grep 'DM' | head -1 | awk '{print$2}')

echo Pulsar: $PSR , parfile: $parfile , DM: $DM
echo_time "Pulsar: $PSR , parfile: $parfile , DM: $DM" >> $obs_logfile 
# --------------- DM from parfile ------------------

DATE=$(date '+%d_%m_%Y')

# Observation Trigger 
#cd /home/dsp/PDR_acquire_setup/PSR_Obs
#scp -pr Trigger.txt vela@172.17.20.222:~/BOOKEEP/.

ssh -X vela@172.17.20.222 "touch ~/BOOKEEP/Trigger.txt"

echo Trigger sent to obs. PC [VELA].
echo_time "Trigger sent to obs. PC [VELA]." >> $obs_logfile 
# count number of file 119
#cd /data/TRANSIENT_BUFFER

# -------------------------
i="0"
#testIfNFiles.sh

ACQDFILES=`ls $Transient_BUFF | wc -l`; # echo Acquired $ACQDFILES
EXPECTEDFILES=$Nfiles;

while [ $ACQDFILES -lt $EXPECTEDFILES ]
    do
    echo -ne ... Not yet over \\r
    sleep 1
    ACQDFILES=`ls $Transient_BUFF | wc -l`; echo -ne Acquired $ACQDFILES
    i=$[$i+1]
    done

echo ACQ over
echo_time "Acquired $ACQDFILES." >> $obs_logfile 
echo_time "ACQ over." >> $obs_logfile 

# Removing Trigger file from remote machine
echo Removing Trigger file from remote machine
echo_time "Removing Trigger file from remote machine " >> $obs_logfile 
ssh -X vela@172.17.20.222 "rm ~/BOOKEEP/Trigger.txt"

#------------------------

echo Observation stoped 
echo_time "Observation stoped." >> $obs_logfile 
date 

ls -ltrh $Transient_BUFF/*  >> $obs_logfile 

#------------------------
# RAW data
raw_data_dir=$parent_dir/$PSR/$DATE
mkdir -p $raw_data_dir

mv $Transient_BUFF/* $raw_data_dir/.

echo_time "RAW PCAP data moved from Transient Buffer." >> $obs_logfile 

# SLIP check
echo_time "SLIP check Started ..." >> $obs_logfile 
cd $raw_data_dir
python3 $slip_check_script $raw_data_dir $obs_logfile

echo_time "SLIP check Finished..." >> $obs_logfile
# mkdir 

processing_dir=$processed_data/$PSR/$DATE
mkdir -p $processing_dir
cd $processing_dir
echo_time "Pulsar data reduction pipeline started ..." >> $obs_logfile
python3 $pipeline_script $raw_data_dir $PSR $CH $phase_file $Freq $obs_logfile

# Remove 2GB PCAP
#echo " "
#echo Removing Raw PCAP files from ... rm -r $raw_data_dir/*
#rm -r $raw_data_dir/*

# Remove DADA file
#echo " "
#DADAFILE=`ls *.dada`
#echo Removing DADA file from ... rm $DADAFILE
#echo_time "Removing DADA file from ... rm $DADAFILE ..." >> $obs_logfile
#rm $DADAFILE

# PRESTO
echo " "

singularity_container='singularity exec /home/dsp/singularity_imag/pschive_py3.sif  '

echo PRESTO
echo_time "--- PRESTO ---" >> $obs_logfile
echo singularity_container: $singularity_container
echo_time "singularity_container: $singularity_container" >> $obs_logfile

FITSFILE=`ls *intensity.fits`
echo Filterbank to Process for PULSAR: $PSR with DM:$DM : $FITSFILE
echo_time "Filterbank to Process for PULSAR: $PSR with DM:$DM : $FITSFILE" >> $obs_logfile
echo " "
echo_time "--- RFIFIND ---" >> $obs_logfile
echo RFIFIND: $singularity_container rfifind -time 2.0 -o $PSR $FILFILE
echo_time "RFIFIND: $singularity_container rfifind -time 2.0 -o $PSR $FITSFILE" >> $obs_logfile
$singularity_container rfifind -time 2.0 -o $PSR $FITSFILE

# Change telescope name
INFFILE=`ls *.inf`
sed -i -e 's/GMRT/GBD_DART/g' $INFFILE

rfiMASK=$PSR'_rfifind.mask'
echo " "
echo_time "--- PREPDATA ---" >> $obs_logfile
echo PREPDATA: $singularity_container prepdata -nobary -o $FITSFILE -dm $DM -mask $rfiMASK $FITSFILE
echo_time "PREPDATA: $singularity_container prepdata -nobary -o $FITSFILE -dm $DM -mask $rfiMASK $FITSFILE" >> $obs_logfile
$singularity_container prepdata -nobary -o $FITSFILE -dm $DM -mask $rfiMASK $FITSFILE

# Change telescope name
INFFILE=`ls *.inf`
sed -i -e 's/GMRT/GBD_DART/g' $INFFILE

echo " "
echo_time "--- REALFFT ---" >> $obs_logfile
echo REALFFT: $singularity_container realfft  $FITSFILE.dat
echo_time "REALFFT: $singularity_container realfft  $FITSFILE.dat" >> $obs_logfile
$singularity_container realfft  $FITSFILE.dat

echo " "
echo_time "--- ACCELSEARCH ---" >> $obs_logfile
echo ACCELSEARCH: $singularity_container accelsearch -numharm 8 -zmax 8  $FITSFILE.fft
echo_time "ACCELSEARCH: $singularity_container accelsearch -numharm 8 -zmax 8  $FITSFILE.fft" >> $obs_logfile
$singularity_container accelsearch -numharm 8 -zmax 8  $FITSFILE.fft


echo_time "Observation Over." >> $obs_logfile
#FIL_file=$(find . -name '*.fil')


#N1=`ls /data/TRANSIENT_BUFFER/ | head -n 1 | cut -c 1-49`
#EXT=`.PCAP_PDR`

#FileName=$N1$EXT

#mkdir GULPED
#TWOGFILE=`ls *_patched.PCAP_PDR` GULPED/.
#mv *_patched.PCAP_PDR GULPED/.
#TWOGFILE=`ls *.sh`


