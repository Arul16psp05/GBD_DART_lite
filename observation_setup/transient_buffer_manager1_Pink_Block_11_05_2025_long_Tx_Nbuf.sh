#!/bin/bash

TRANSIENTBUF=$1 #'/home/gbdobserver/Ramdisk_70GB'

INTERVAL=0.05
FILELIST=$2/filelist_pink_box.$(date +%s)    #/tmp/filelist.$(date +%s) #$TARGET/
TRIGGER=$3 #~/BOOKEEP/Trigger.txt
DESTINATION=$4:$5 #summer@172.17.20.227:/home/summer/TRANSIENT_BUFFER_200GB
TEST_FOR_FILE=$6 #/home/gbdobserver/PDR_acquire_setup/testForFile.sh

cd $TRANSIENTBUF

x=1
NFILES2KEEP=$7 #32  
#NUM_FILES_TX=$8  # from Argument
CURRENT_TX=0

echo Number of files to keep in Pink buffer = $NFILES2KEEP
echo Nubser of files to transfer from the Pink buffer = 

while [ $x -le 5 ]
do

    ls -ltrh *.PCAP_PDR | basename -a  $(awk '{print $9}') > $FILELIST
    
    if [ $(cat $FILELIST | wc -l) -ge $NFILES2KEEP ]
    then
       file2Delete=`ls -ltRh | tail -n 1 | awk '{print $9}'`; echo $file2Delete
       echo -n removing the oldest file $file2Delete   ... 
       rm -rf $file2Delete
       echo done    
    fi
    # -------------------------------------------------------------------------------------
    
    # https://www.geeksforgeeks.org/bash-scripting-how-to-check-if-file-exists/
    
    #bash $TEST_FOR_FILE $TRIGGER  
    #ans=$(echo $?)
    
    #if [ $ans -eq 0 ] 
    
    if test -f $TRIGGER ;
    then
       
       CURRENT_BATCH_FILES=$(cat $FILELIST | wc -l)
       echo CURRENT_BATCH_FILES =  $CURRENT_BATCH_FILES ,  CURRENT_TX = $CURRENT_TX
       
       CURRENT_TX=$(($CURRENT_TX+$CURRENT_BATCH_FILES))
       echo CURRENT_TX = $CURRENT_TX 
        
       echo Will RSYNC following files to remote-Tower/GBD machine
       cat $FILELIST | wc -l
       cat $FILELIST
       echo
       
       for File in $(cat $FILELIST) ; do
          #echo $File
          #rsync -avP --bwlimit=80M --remove-source-files  $TRANSIENTBUF/$File $DESTINATION # Introduce --bw-limit=??M 
          rsync -avP --remove-source-files  $TRANSIENTBUF/$File $DESTINATION
       done
       
    else
       CURRENT_TX=0
       echo --  no trigger found  -- - 
       sleep 5 
    fi

done


