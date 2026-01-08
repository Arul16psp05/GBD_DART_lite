
import numpy as np
import datetime, os, sys, glob, shutil, logging 


parent_dir   = sys.argv[1] # "/data/dsp/GBD_TRANSIENT_BUFFER/1PPS/test"  
obs_logfile  = sys.argv[2] # Observation log file

#------------ Logging ------------------
# https://realpython.com/python-logging/
logging.basicConfig(filename=obs_logfile, filemode="a", format="{asctime} - {levelname} - {message}", style="{", datefmt="%d-%m-%Y %H:%M",)
#------------ Logging ------------------


files = list(filter(os.path.isfile, glob.glob(parent_dir +'/'+ "*.PCAP_PDR")))
files.sort(key=lambda x: os.path.getmtime(x))

logging.warning("Data Directory: "+ str(parent_dir))
logging.warning("Number of Files: "+ str(len(files)))

nfft 			= 512			# Change if nessesory
max_nfft_per_pkt 	= 512	

PCAP_Global_HDR	= 24
PCAP_pkt_HDR		= 16
Network_HDR     	= 42	
MBR_pkt_HDR		= 32
data_len 		= 1024

udp_pkt_len		= (PCAP_pkt_HDR + Network_HDR + MBR_pkt_HDR + data_len)
HDR_len 		= udp_pkt_len - data_len

udp_pkt_len_w_GHDR	= udp_pkt_len + PCAP_Global_HDR
HDR_len_w_GHDR 	= HDR_len + PCAP_Global_HDR

def meta_data(header):
    #  This value is in seconds since January 1, 1970 00:00:00 GMT
    ts_sec 	= int.from_bytes(header[:4], byteorder='little')
    ts_usec	= int.from_bytes(header[4:8], byteorder='little')
    pkt_Tstamp  = np.datetime64(ts_sec, 's') + np.timedelta64(ts_usec, 'us')   
    pkt_count   = int.from_bytes(header[-4:], byteorder='big')
    gps_count   = int.from_bytes(header[-6:-4], byteorder='big')
    return pkt_count, gps_count, pkt_Tstamp

# make folder to keep the damaged files
gulped_dir = parent_dir+'/GULPED'
os.mkdir(gulped_dir)
         
for file_n in files:
    crnt_file_fid = open(file_n,'rb')
    crnt_file_fid.seek(PCAP_Global_HDR)
    
    file_size = os.path.getsize(file_n)
    
    pkt = np.fromfile(crnt_file_fid, dtype='int8', count=udp_pkt_len, sep='')    
    header = pkt[:HDR_len]
    
    first_pkt_count, first_gps_count, first_pkt_Tstamp = meta_data(header)
    

    last_pkt = (file_size - udp_pkt_len)
    crnt_file_fid.seek(last_pkt)
    pkt = np.fromfile(crnt_file_fid, dtype='int8', count=udp_pkt_len, sep='')    
    header = pkt[:HDR_len]
    last_pkt_count, last_gps_count, last_pkt_Tstamp = meta_data(header)    
    
    crnt_file_fid.close()
    
    if (((file_size - PCAP_Global_HDR) / udp_pkt_len) - (last_pkt_count - first_pkt_count)) == 1 :
        print("No packet loss")
        logging.warning("No packet loss: "+ str(file_n))
    else:
        logging.warning("Packet loss: "+ str(file_n))
        print( "Packet loss found .... Being fixed ... Please wait ..." )
        logging.warning("Packet loss found .... Being fixed ... Please wait ..." )       
        # Patch
        
        filename = file_n 
        
        file_size = os.path.getsize(filename)
        No_of_pkts = int((file_size - PCAP_Global_HDR) / udp_pkt_len)

        # ------------------- PCAP data issue ----------------------
        fraction_pkts = ((file_size - PCAP_Global_HDR) / udp_pkt_len) - No_of_pkts
        skip_bytes = int(fraction_pkts * udp_pkt_len)
        # ------------------- PCAP data issue ----------------------
        # date_time_for out file
        #https://dev.to/gagangulyani/creating-files-with-python-and-mimicking-the-touch-command-hhm

        st_info = os.stat(filename)
        access_time = st_info.st_atime 
        modified_time = st_info.st_mtime
        #created_time = st_info.st_ctime        
        
        temp_fname = filename+'.pcap'
        shutil.move(filename, temp_fname)
        
        new_name = ((filename.rsplit('/')[-1]).rsplit('.',1))[0]
        out_new_name = (new_name+'_filtered.PCAP_PDR.pcap')
                
        filter_cmd = 'tcpdump -r '+ temp_fname + ' -w ' + out_new_name + '  host 172.10.10.1 and port 55000 ' 
        
        print('PCAP filtering ... ' +  filter_cmd) 
        logging.warning('PCAP filtering ... ' +  filter_cmd)         
        os.system(filter_cmd)
        
        out_file_name = ((out_new_name.rsplit('/')[-1]).rsplit('.',1))[0]
        
        shutil.move(out_new_name, out_file_name)
                

        # Change time
        os.utime(out_file_name, (access_time, modified_time))
        print('Patched ' + str(file_n)+ ' as ' + str(out_file_name))
        logging.warning('Patched ' + str(file_n)+ ' as ' + str(out_file_name))        

        shutil.move((filename+'.pcap'), filename) 
        shutil.move(filename, gulped_dir)        
        
        print('Gulped file moved to ' +  gulped_dir)       
        logging.warning('Gulped file moved to ' +  gulped_dir) 
                
    print("first_pkt_count, first_gps_count, first_pkt_Tstamp")
    print(first_pkt_count, first_gps_count, first_pkt_Tstamp)
    print("last_pkt_count, last_gps_count, last_pkt_Tstamp")
    print(last_pkt_count, last_gps_count, last_pkt_Tstamp) 
    print("diff") 
    print(last_pkt_count - first_pkt_count, last_gps_count - first_gps_count, last_pkt_Tstamp - first_pkt_Tstamp)  
    logging.warning("first_pkt_count, first_gps_count, first_pkt_Tstamp" )
    logging.warning(str(first_pkt_count) +" "+ str(first_gps_count) +" "+ str(first_pkt_Tstamp))   
    logging.warning("last_pkt_count, last_gps_count, last_pkt_Tstamp" )
    logging.warning(str(last_pkt_count) +" "+ str(last_gps_count) +" "+ str(last_pkt_Tstamp))   
    logging.warning('diff_pkt_count  , diff_gps_count ,  diff_pkt_timstamp')
    logging.warning(str(last_pkt_count - first_pkt_count) +" "+ str(last_gps_count - first_gps_count)+" "+ str(last_pkt_Tstamp - first_pkt_Tstamp))    
    print("")

#         tcpdump -r PDR_GULP_data002.pcap_12_05_2025_14_26_1747040174.PCAP_PDR.pcap -w PDR_GULP_data002.pcap_12_05_2025_14_26_1747040174.PCAP_PDR_fixed.pcap host 172.10.10.1 and port 55000    
