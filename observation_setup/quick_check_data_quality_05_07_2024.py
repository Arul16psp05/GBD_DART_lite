import matplotlib.pyplot as plt
import numpy as np
#from astropy import time as T, units as u
from datetime import timezone 
import datetime, os, sys

filename_w_filepath 	= sys.argv[1] #
#filename_w_filepath 	= 'lab_test001.pcap_28_03_2024_12_00_1711607414.PCAP_PDR.test'

psrname = "B1929+10"

number_of_pkts = int(sys.argv[2]) #10

nfft 			= 512			# Change if nessesory
#nsamples_per_pkt_per_channal = nfft
max_nfft_per_pkt 	= 512	

PCAP_Global_HDR		= 24
PCAP_pkt_HDR		= 16
Network_HDR     	= 42	
MBR_pkt_HDR		= 32
data_len 		= 1024

udp_pkt_len		= (PCAP_pkt_HDR + Network_HDR + MBR_pkt_HDR + data_len)
HDR_len 		= udp_pkt_len - data_len

udp_pkt_len_w_GHDR	= udp_pkt_len + PCAP_Global_HDR
HDR_len_w_GHDR 		= HDR_len + PCAP_Global_HDR

file_size 		= os.path.getsize(filename_w_filepath)

num_packets 		= int((file_size - PCAP_Global_HDR) / udp_pkt_len)
num_spectrum 		= int(num_packets / (nfft / max_nfft_per_pkt))

obs_time     		= (num_packets * nfft)/33e6

# -------------------------------- Functions -----------------------------------------

def perform_fft(raw_data):

    fft_data = np.fft.fft(raw_data)
    return fft_data

def cross_correlation(fft_data_1, fft_data_2):

    cc_data = fft_data_1 * np.conj(fft_data_2)
    return cc_data

def perform_ifft(cc_data):
    
    Tseries_data = np.fft.ifft(cc_data)
    return Tseries_data

def power(fft_data):

    pwr_spec = np.abs(fft_data)**2
    return pwr_spec
    
# -------------------------------- Functions -----------------------------------------

crnt_file_fid = open(filename_w_filepath,'rb')
crnt_file_fid.seek(PCAP_Global_HDR)

channal_1 = np.zeros(int(number_of_pkts*nfft), dtype=np.int8)
channal_2 = np.zeros(int(number_of_pkts*nfft), dtype=np.int8)


for pkt_idx in range(number_of_pkts):
	
    raw  		= np.fromfile(crnt_file_fid, dtype='int8', count=udp_pkt_len, sep='')

    hdr  		= raw[:HDR_len]
    data 		= raw[HDR_len:] 
    
    ch_1 		= data[::2]
    ch_2 		= data[1::2]
    
    channal_1[pkt_idx*nfft:(pkt_idx*nfft)+nfft] = ch_1
    channal_2[pkt_idx*nfft:(pkt_idx*nfft)+nfft] = ch_2
    
fft_data_ch1   	= perform_fft(ch_1)
fft_data_ch2   	= perform_fft(ch_2) 

std_ch1 = np.std(channal_1)
std_ch2 = np.std(channal_2)

    #correlation    	= cross_correlation(fft_data_ch1, fft_data_ch2)
    
    #CC_Tseries 		= perform_ifft(correlation/64)
    
    
#fig.suptitle('This is a somewhat long figure title', fontsize=16)
plt.suptitle(str(int(number_of_pkts*nfft)) +' Samples', fontsize=16)
    
plt.subplot(3,2,1)
plt.plot(channal_1, '.')
plt.legend(['ch_1'])

plt.subplot(3,2,2)
plt.plot(channal_2 , '.')
plt.legend(['ch_2'])

plt.subplot(3,2,3)
plt.title('STD: '+ str(round(std_ch1)) + '---' + '5_STD: '+ str(round(5*std_ch1)))
plt.hist(channal_1, 100)
plt.legend(['ch_1'])

plt.subplot(3,2,4)
plt.title('STD: '+ str(round(std_ch2)) + '---' + '5_STD: '+ str(round(5*std_ch2)))
plt.hist(channal_2, 100)
plt.legend(['ch_2'])

plt.subplot(3,2,5)
plt.plot(10*np.log10(power(fft_data_ch1[int(nfft/2):])))
plt.legend(['ch_1'])

plt.subplot(3,2,6)
plt.plot(10*np.log10(power(fft_data_ch2[int(nfft/2):])))
plt.legend(['ch_2'])

plt.show()
