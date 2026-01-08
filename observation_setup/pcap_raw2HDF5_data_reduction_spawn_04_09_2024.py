#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 23:26:59 2023

@author: arul
"""
import glob, os, sys, shutil
import time, datetime, threading
import numpy as np

dir_name                = str(sys.argv[1]) # "/home/gbdobserver/Ramdisk_20GB" 
data_reduction_script   = sys.argv[2]
phase_cal_file          = sys.argv[3]
Archive_dir_HDF5        = str(sys.argv[4]) # "/data/PDR_data_archive"


print("Script to be spawned: "+ str(data_reduction_script))
print("Cal file: "+ str(phase_cal_file))

# ----------------------Basic Packet info ---------------------

nfft 		 = 512			# Change if nessesory
max_nfft_per_pkt = 512	
Time_avg       	 = 0.5   # in secs

PCAP_Global_HDR	= 24
PCAP_pkt_HDR	= 16
Network_HDR     = 42	
MBR_pkt_HDR	= 32
data_len 	= 1024

udp_pkt_len	= (PCAP_pkt_HDR + Network_HDR + MBR_pkt_HDR + data_len)
HDR_len 	= udp_pkt_len - data_len

udp_pkt_len_w_GHDR	= udp_pkt_len + PCAP_Global_HDR
HDR_len_w_GHDR 		= HDR_len + PCAP_Global_HDR

# ----------------------Basic Packet info ---------------------

# ------------------- Directory creation ----------
#Archive_dir = "/media/vela/cc6f4845-ca6b-47bd-b0ab-44291550fe36/data"
Archive_dir = Archive_dir_HDF5 
data_dir = os.path.join(Archive_dir, datetime.datetime.now().strftime('%d_%m_%Y_%H_%M'))
os.makedirs(data_dir)
os.chdir(data_dir)

# ------------------- Directory creation ----------


def update(dir_name):
    file_name = glob.glob(dir_name + "/*.PCAP_PDR")   
    file_name.sort(key=os.path.getmtime)
    return file_name
    
#file_name = np.loadtxt(filename, dtype='str') 

while True:
    file_name = update(dir_name)
    if len(file_name) == 0:
        print(" Waiting for Data files ... ", end='\r')
    else:
        break
    time.sleep(5)


def temp_chunk_writer(file, temp_w_filepath, check_temp_file, file_no,  file_size, num_packets, num_spectrum, N_avg):
    
    if check_temp_file is True:
        print('Check_temp_file: Yes')
    
        temp_file_size 		= os.path.getsize(temp_w_filepath)
        temp_file_num_packets 	= temp_file_size//udp_pkt_len
        temp_file_num_spectrum 	= int(temp_file_num_packets / (nfft / max_nfft_per_pkt))
        num_spectrum 		= num_spectrum + temp_file_num_spectrum
        num_packets		= num_packets + temp_file_num_packets
        temp_file_obs_time 	= float(temp_file_num_packets * nfft)/33e6
        temp_file_fid 		= open(temp_w_filepath,'rb')
    else:
        print('Check_temp_file: No')
        temp_file_obs_time = 0

    num_outerloop  		= int(num_spectrum / N_avg)

    spec2temp_file  = int ( num_packets - (num_outerloop * N_avg) )

    bytes2temp_file = int( spec2temp_file * udp_pkt_len )

    pointer = file_size - bytes2temp_file

    crnt_file_fid = open(file,'rb')

    crnt_file_fid.seek(pointer)
    
    temp_w_filepath  = dir_name + "/temp_pdr" + str(file_no)+".temp"
    
    out_temp_file = open(temp_w_filepath,'wb')
    temp_data = np.fromfile(crnt_file_fid, dtype='int8', count = bytes2temp_file, sep='')
    temp_data.tofile(out_temp_file, sep='')
    out_temp_file.close()
    return temp_w_filepath


def main_file_reader(file_name, nfft=nfft, max_nfft_per_pkt=max_nfft_per_pkt, PCAP_Global_HDR=PCAP_Global_HDR, udp_pkt_len=udp_pkt_len, Time_avg=Time_avg):

    temp_w_filepath = dir_name + '/temp_MBR.temp'

    temp_PDR_data = np.array([], dtype='str') 
    temp_PDR_data = np.append(temp_PDR_data, temp_w_filepath )

    for file_no in range(len(file_name)):

        file 		= file_name[file_no]
        file_size 	= os.path.getsize(file_name[file_no])

        num_packets 	= int((file_size - PCAP_Global_HDR) / udp_pkt_len)
        num_spectrum 	= int(num_packets / (nfft / max_nfft_per_pkt))

        obs_time     	= (num_packets * nfft)/33e6

        N_avg          	= int((num_spectrum/obs_time) * Time_avg)
    
        check_temp_file = os.path.isfile(temp_w_filepath)
    
    
        temp_w_filepath = temp_chunk_writer(file, temp_w_filepath, check_temp_file, file_no,  file_size, num_packets, num_spectrum, N_avg)
    
        temp_PDR_data = np.append(temp_PDR_data, temp_w_filepath )
    
        time.sleep(0.01)
    return temp_PDR_data, N_avg

def data_reduction_exec(filename, temp_PDR_data, N_avg):
        exec_cmd = "python3 "+ str(data_reduction_script)+" "+ str(filename) +" "+ str(temp_PDR_data) +" "+ str(N_avg) + " " + str(phase_cal_file)
        #os.system(exec_cmd)    
        print(exec_cmd)
        os.system(exec_cmd)

def threadpool_executer(file_name, temp_PDR_data, N_avg, data_dir):

    for file_nom in range(len(file_name)):
        globals()['threadobj%s' % file_nom] = threading.Thread(target=data_reduction_exec, args=(str(file_name[file_nom]), str(temp_PDR_data[file_nom]), str(N_avg),), daemon=True)
    
    for file_nom in range(len(file_name)):
        globals()['threadobj%s' % file_nom].start()
        time.sleep(0.3)
        
    print("Thread initiated ..... ")
    
    for file_nom in range(len(file_name)):
        globals()['threadobj%s' % file_nom].join()
    
    print("Thread finished ..... ")
    
    os.rename(temp_PDR_data[-1], temp_PDR_data[0])
    
    for file_nom in range(len(file_name)):
        old_fname = data_dir+'/'+(file_name[file_nom].rsplit('/')[-1])+'_'+str(N_avg)+'_Spec_Avg_allX.hdf5'
        new_fname = data_dir+'/'+(file_name[file_nom].rsplit('/')[-1])+'_'+str(file_nom)+'_'+str(N_avg)+'_Spec_Avg_allX.hdf5'
        
        shutil.copy(old_fname, new_fname)
        time.sleep(0.1)
        os.remove(old_fname)
print(file_name)
temp_PDR_data, N_avg = main_file_reader(file_name) 
threadpool_executer(file_name, temp_PDR_data, N_avg, data_dir)
old_files = file_name

#old_files = data_reduction_exec(file_name, temp_PDR_data, N_avg)

#return old_files

while True:
    file_name = update(dir_name)
    new_files = [x for x in file_name if x not in old_files]
    print(new_files)
    if len(new_files) != 0:
        old_files = np.append(old_files, new_files)
        file_name = new_files
        temp_PDR_data, N_avg = main_file_reader(file_name) 
        threadpool_executer(file_name, temp_PDR_data, N_avg, data_dir)
        old_files = file_name
        
    time.sleep(5)
    
