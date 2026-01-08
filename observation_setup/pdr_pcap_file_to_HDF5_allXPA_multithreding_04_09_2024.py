"""
# python3

This script to read and process MPPR and PDR voltage data
"""

import os, sys, h5py
import numpy as np
from progress.bar import IncrementalBar

filename_w_filepath 	= sys.argv[1] #

temp_w_filepath 	= sys.argv[2] #base_path + '/temp_MBR.pcap'

N_avg			    = int(sys.argv[3])

phase_cal_file_name = sys.argv[4]     # Phase data file ( Deg. )

check_temp_file = os.path.isfile(temp_w_filepath)

nfft 		= 512			# Change if nessesory
max_nfft_per_pkt = 512	

PCAP_Global_HDR	= 24
PCAP_pkt_HDR	= 16
Network_HDR     = 42	
MBR_pkt_HDR	= 32
data_len 	= 1024

udp_pkt_len	= (PCAP_pkt_HDR + Network_HDR + MBR_pkt_HDR + data_len)
HDR_len 	= udp_pkt_len - data_len

udp_pkt_len_w_GHDR	= udp_pkt_len + PCAP_Global_HDR
HDR_len_w_GHDR 		= HDR_len + PCAP_Global_HDR

file_size 	= os.path.getsize(filename_w_filepath)

num_packets 	= int((file_size - PCAP_Global_HDR) / udp_pkt_len)
num_spectrum 	= int(num_packets / (nfft / max_nfft_per_pkt))

obs_time     	= (num_packets * nfft)/33e6

out_data_file_name = (filename_w_filepath.rsplit('/')[-1])+'_'+str(N_avg)+'_Spec_Avg_allX.hdf5'
hdf5_file = h5py.File(out_data_file_name, 'a')


# -------------------------------- Functions -----------------------------------------

def perform_fft(raw_data):

    fft_data = np.fft.rfft(raw_data)[1:]
    return fft_data

def power(fft_data):

    pwr_spec = np.abs(fft_data)**2
    return pwr_spec


def cross_correlation(fft_data_1, fft_data_2):

    cc_data = fft_data_1 * np.conj(fft_data_2)
    #cc_data = fft_data_1 * np.conj(fft_data_1)
    #cc_data = fft_data_2 * np.conj(fft_data_2)
    return cc_data

def addition(fft_data_1, fft_data_2):

    add_data = np.add(power(fft_data_1), power(fft_data_2))
    return add_data

# -------------------------------- Functions -----------------------------------------

# -------------------------------- Phase Cal -----------------------------------------

cal_phase_data = np.loadtxt(phase_cal_file_name, dtype="f16", delimiter=",")

rotate = 1j**(cal_phase_data/90) 		# Phase rotation

cal_phase = rotate

# -------------------------------- Phase Cal -----------------------------------------

print("Total number of packets in this file: "+ str(num_packets))

if check_temp_file is True:
    print('Check_temp_file: Yes')
    
    temp_file_size 		= os.path.getsize(temp_w_filepath)
    temp_file_num_packets 	= temp_file_size//udp_pkt_len
    temp_file_num_spectrum 	= int(temp_file_num_packets / (nfft / max_nfft_per_pkt))
    num_spectrum 		= num_spectrum + temp_file_num_spectrum
    num_packets			= num_packets + temp_file_num_packets
    temp_file_obs_time 		= float(temp_file_num_packets * nfft)/33e6
    temp_file_fid 		= open(temp_w_filepath,'rb')
else:
    print('Check_temp_file: No')
    temp_file_obs_time = 0

num_outerloop  		= int(num_spectrum / N_avg)

spec2temp_file  = int ( num_packets - (num_outerloop * N_avg) )

bytes2temp_file = int( spec2temp_file * udp_pkt_len )

#out_temp_file_pkts 	= int(num_packets - (num_outerloop * N_avg) )

# ----------------- for processing bar ------------------------

class FancyBar(IncrementalBar):
    message 	= 'Loading'
    fill 	= '*'
    suffix 	= '%(percent).1f%% - %(eta)ds / %(elapsed)ds Elapsed'  #   
    
bar = FancyBar('MBR2HDF5 convertion ', max=int(num_outerloop))		

# ----------------- for processing bar ------------------------

crnt_file_fid = open(filename_w_filepath,'rb')

crnt_file_fid.seek(PCAP_Global_HDR)

corr_spec_avg = np.empty( shape = [num_outerloop, int(nfft/2)], dtype = complex )

acorr_spec_avg1 = np.empty(shape=[num_outerloop, int(nfft/2)], dtype = complex)
acorr_spec_avg2 = np.empty(shape=[num_outerloop, int(nfft/2)], dtype = complex)

p_sum_avg = np.empty(shape=[num_outerloop, int(nfft/2)], dtype = float)

for kk in range(num_outerloop):  
   
    corr_spec = np.empty( shape = [N_avg, int(nfft/2)], dtype = complex ) 
    
    acorr_spec1 = np.empty(shape=[N_avg, int(nfft/2)], dtype = complex) 
    acorr_spec2 = np.empty(shape=[N_avg, int(nfft/2)], dtype = complex) 
      
    p_sum = np.empty(shape=[N_avg, int(nfft/2)], dtype = float)
   
    for idx in range(N_avg):
        
        if (kk == 0 and check_temp_file == True and idx <= temp_file_num_spectrum-1):    
            #print("Reading temp file ... ")  
            raw  = np.fromfile(temp_file_fid, dtype='int8', count=udp_pkt_len, sep='')
            hdr  = raw[:HDR_len]
            data = raw[HDR_len:] 

        else:    
            #print("Reading current file ... ")          
            raw  = np.fromfile(crnt_file_fid, dtype='int8', count=udp_pkt_len, sep='')
            hdr  = raw[:HDR_len]
            data = raw[HDR_len:]     
        
        # ---------------- PKT check ------------------------        
        pkt_count = int.from_bytes(hdr[-4:], byteorder='big')
        #print(pkt_count)
        if kk == 0 and idx == 0:
            previous_pkt = pkt_count
            missed_pkt   = 0
            print("First Packet:      "+ str(pkt_count))
            
        # prev_pkt = check_pkt_count(pkt_count, previous_pkt)
        # previous_pkt = prev_pkt
        # ---------------- PKT check ------------------------
        ch_1 = data[::2]
        ch_2 = data[1::2]
        
        fft_data_ch1   = perform_fft(ch_1)
        fft_data_ch2   = perform_fft(ch_2)

        correlation    = cross_correlation(fft_data_ch1, fft_data_ch2)
        
        v_sum = fft_data_ch1 + (fft_data_ch2 * cal_phase)
        
        p_sum[idx] = (np.abs(v_sum))**2 

        acorrelation1 = cross_correlation(fft_data_ch1, fft_data_ch1) ; 
        acorrelation2 = cross_correlation(fft_data_ch2, fft_data_ch2) ;
        
        corr_spec[idx] = correlation
        
        acorr_spec1[idx] = acorrelation1
        acorr_spec2[idx] = acorrelation2
        
        
    corr_spec_avg[kk]  = np.mean(corr_spec, axis=0)   
    
    acorr_spec_avg1[kk] = np.mean(acorr_spec1, axis=0)   
    acorr_spec_avg2[kk] = np.mean(acorr_spec2, axis=0)  
   
    p_sum_avg[kk] = np.mean(p_sum, axis=0)

    bar.next()									# Progress bar
bar.finish()

print("Last packet count: "+ str(pkt_count))
start_time = os.path.getmtime(filename_w_filepath)
timestamp  = np.linspace((start_time-temp_file_obs_time), (start_time+obs_time), num_outerloop)

# Create the dataset at first
hdf5_file.create_dataset('corr_spec_avg', data=corr_spec_avg)

hdf5_file.create_dataset('acorr_spec_avg1', data=acorr_spec_avg1)
hdf5_file.create_dataset('acorr_spec_avg2', data=acorr_spec_avg2)

hdf5_file.create_dataset('p_sum_avg', data=p_sum_avg)

hdf5_file.create_dataset('timestamp', data=timestamp, dtype='float')

print("Total number of Spectra written to the file: "+ str(num_outerloop))

if check_temp_file is True:
    temp_file_fid.close()
    os.remove(temp_w_filepath)

os.remove(filename_w_filepath)
