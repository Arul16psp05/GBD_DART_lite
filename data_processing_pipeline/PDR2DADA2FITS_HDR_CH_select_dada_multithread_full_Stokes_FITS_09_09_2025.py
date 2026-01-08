"""
This python script will covert PDR data to DADA format for Pulsar data processing using DSPSR.

by Arul @ 06_04_2024

USAGE: python3 < pulsar data directory path > < pulsar Jname > <CH_sel> <Phase file> <170> <log_file.log>

python3 /home/dsp/PDR_acquire_setup/dada_scripts/PDR2DADA2FITS_HDR_CH_select_dada_multithread_full_Stokes_FITS_09_09_2025.py /data/dsp/GBD_TRANSIENT_BUFFER/J0534+2200 J0534+2200 0 /home/dsp/PDR_acquire_setup/dada_scripts/phase/rf_path_phase.dat 175 /home/dsp/PDR_acquire_setup/master_obs_log/J0534+2200_11_05_2025_observation.log

###############################

Modifications:
Line number   |     Note
48, 160       | creation to modified time - os.path.getmtime

03.05.2025 - Arul

PDR2DADA2FITS_HDR_CH_select_multithread_with_FIL_file_v2_noSPECflip_obsFREQ_15_01_2025.py > PDR2DADA2FITS_HDR_CH_select_multithread_with_FIL_03_05_2025.py

1) Iterative cleaner added, performs before goes to PDMP
2) Reduced Filterbank is available - Pulsar specific --> Nspec_AVG = (P0/nbin)/Tsamp
3) PSRFITS outfile modes available - (a) coherent dedispersion, (b) Incoherent dedispersion 
4) Multithread enabled in (a) DSPSR, (b) DIGIFIL
5) CPU cores useage increesed to 70 percentage

12.08.2025 - Arul

PDR2DADA2FITS_HDR_CH_select_multithread_with_FIL_03_05_2025.py --> PDR2DADA2FITS_HDR_CH_select_multithread_full_Stokes_FITS_12_08_2025.py

1) DIGIFITS used to convert - DADA to Serach mode FITS
2) Full stokes mode - a) Search mode b) Folded profile
3) Search mode spectrum resolution, CRAB - 128 uS & other 4 pulsars - 256 uS 
4) filterbank files not available in this mode

09.09.2025 - Arul

PDR2DADA2FITS_HDR_CH_select_multithread_full_Stokes_FITS_12_08_2025.py --> PDR2DADA2FITS_HDR_CH_select_dada_multithread_full_Stokes_FITS_09_09_2025.py

1) DADA file HEADER issue solved - MODE: from PSR to SURVAY

###############################

"""

#import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time
from datetime import timezone 
import datetime, os, sys, glob, logging
#from multiprocessing import Process, Queue
import multiprocessing as mp
    
RFI_mit_pyscript = '/home/dsp/PDR_acquire_setup/iterative_cleaner.py'

percentage_of_cores_to_use 	= 70 # In percentage
no_of_cores			= round(percentage_of_cores_to_use*(os.cpu_count()/100))

start_time = datetime.datetime.now()
print(start_time)

#parent_dir = '/home/dsp/Transient_data'

parent_dir        = sys.argv[1] 
psrname           = sys.argv[2]   # PSRJ name
CH_sel               = sys.argv[3]   # 1: channel-1 , 2: channel-2, 4: crosscorrelation of Channel 1 and 2
Phase_file        = sys.argv[4]
obs_freqency = sys.argv[5]   # In MHz
obs_logfile       = sys.argv[6] # Observation log file

#------------ Logging ------------------
# https://realpython.com/python-logging/
logging.basicConfig(filename=obs_logfile, filemode="a", format="{asctime} - {levelname} - {message}", style="{", datefmt="%d-%m-%Y %H:%M",)
#------------ Logging ------------------

logging.warning("Data Directory: "+ str(parent_dir))
logging.warning("Pulsar: "+ str(psrname))
logging.warning("CH selelction: "+ str(CH_sel))
logging.warning("Phase file: "+ str(Phase_file))
logging.warning("Obs. logfile: "+ str(obs_logfile))


files = list(filter(os.path.isfile, glob.glob(parent_dir +'/'+ "*.PCAP_PDR")))
files.sort(key=lambda x: os.path.getmtime(x))

#files = files[0:10]

filename_w_filepath     = files[0]

fileCtime = datetime.datetime.fromtimestamp(os.path.getmtime(filename_w_filepath)).strftime("%Y_%m_%d_%H_%M_%S")
logging.warning("Observation start time: "+ str(fileCtime))

pardir='/home/dsp/parfiles/'
parfile=str(pardir)+str(psrname)+'.par'

logging.warning("parfile: "+ str(parfile))

#Reading parfile 
def fetch_par_info(parfile_name):
   with open(parfile_name, 'r') as par_file:
      par_lines = par_file.readlines()
      par_tokens = dict([list(filter(lambda x: len(x)>0,line.rsplit()))[:2]for line in par_lines ])
                
      fetch_par_info.dm = par_tokens["DM"]
      fetch_par_info.f0 = par_tokens["F0"]
      if "RAJ" in par_tokens and "DECJ" in par_tokens:
          fetch_par_info.raj = par_tokens["RAJ"]
          fetch_par_info.decj = par_tokens["DECJ"]
          fetch_par_info.RA_DEC = fetch_par_info.raj+fetch_par_info.decj
      else:
          print("[ERROR] Unable to read coordinates from par file. Setting 00:00:00+00:00:00.")
          return "00:00:00","+00:00:00"
          
fetch_par_info(parfile)
dm = str(fetch_par_info.dm)
spin_freq = str(fetch_par_info.f0)
period = str(1/float(spin_freq))
RAJ    = str(fetch_par_info.raj)
DECJ   = str(fetch_par_info.decj)
RA_DEC = str(fetch_par_info.RA_DEC)

logging.warning("DM: "+ str(dm))
logging.warning("Period: "+ str(period))
logging.warning("RA_DEC: "+ str(RA_DEC))

nfft 			= 512			# Change if nessesory
max_nfft_per_pkt 	= 512	

PCAP_Global_HDR	        = 24
PCAP_pkt_HDR		= 16
Network_HDR     	= 42	
MBR_pkt_HDR		= 32
data_len 		= 1024

udp_pkt_len		= (PCAP_pkt_HDR + Network_HDR + MBR_pkt_HDR + data_len)
HDR_len 		= udp_pkt_len - data_len

udp_pkt_len_w_GHDR	= udp_pkt_len + PCAP_Global_HDR
HDR_len_w_GHDR 		= HDR_len + PCAP_Global_HDR

file_size = 0
for file_n in files:
    file_size = file_size + os.path.getsize(file_n)

data_size = int(((file_size - (len(files)*PCAP_Global_HDR) ) / udp_pkt_len ) * (data_len/2))

print('data size: ' + str(data_size*2)+ ' bytes')


out_dada_file = str(psrname)+'_'+str(fileCtime)+'_type_'+str(CH_sel)+'.dada'

# ------------------------------ Compute Phase ---------------------------------------

print("phase_cal_file_name: " + str(Phase_file))

cal_phase_data = np.loadtxt(Phase_file, dtype="f16", delimiter=",")

temp_phase = np.zeros(nfft, dtype="f16")
temp_phase[:256], temp_phase[256:] = -cal_phase_data, cal_phase_data

phase = 1j**(temp_phase/90) 		# Phase rotation

# ------------------------------ Compute Phase ---------------------------------------

# -------------------------------- Functions -----------------------------------------

def perform_fft(raw_data):

    fft_data = np.fft.fft(raw_data)
    return fft_data

def cross_correlation(fft_data_1, fft_data_2, phase=phase):

    cc_data = fft_data_1 * np.conj(fft_data_2*phase)
    return cc_data

def perform_ifft(cc_data):
    
    Tseries_data = np.fft.ifft(cc_data)
    return Tseries_data

#def power(fft_data):
#
#    pwr_spec = np.abs(fft_data)**2
#    return pwr_spec
    
def V_addition(fft_data_1, fft_data_2, phase=phase):

    add_data = np.add(fft_data_1, fft_data_2*phase)
    return add_data
        
# -------------------------------- Functions -----------------------------------------



# --------------------------- HDR section ----------------------------
#https://sourceforge.net/p/psrdada/code/ci/multi_write_bufs/tree/puma2/puma2_header.txt#l29

Center_freq     = str(obs_freqency) #200	#199.75 #	# In MHz
sample_rate	= 33		# In MHz
  
# Getting the current date 
# and time 
#dt = np.datetime64(datetime.datetime.now(timezone.utc))
dt = np.datetime64(datetime.datetime.fromtimestamp(os.path.getmtime(filename_w_filepath), timezone.utc))

HEADER          = 'DADA'
HDR_VERSION 	= '1.0'
HDR_SIZE     	= '4096'
DADA_VERSION    = '1.0'
#FILE_SIZE       = str(data_size)
BW    		= str(np.float64(-sample_rate/2))
FREQ  		= str(Center_freq)
TELESCOPE 	= 'GMRT' #'GBD_PSR_POL_arr'
RECEIVER 	= 'PDR'# 'Fake' #'PDR' # MULTI
INSTRUMENT 	= 'CPSR2' #'Fake' #'GSB'
SOURCE  	= str(psrname)
RA              = str(RAJ)
DEC             = str(DECJ)
MODE            = str('SURVEY')
#MODE  		= 'PSR' # (PSR, CAL, SEARCH)
#STATE      = str('PPQQ')
NBIT 		= str(8)
NCHAN 		= str(1)
NDIM  		= str(1) # real-1, complex-2
NPOL 		= str(2)
NDAT 		= str(data_size) #str(num_outerloop * nfft * len(files))
OBS_OFFSET 	= str(0)
UTC_START  	= str(dt).rsplit('T')
UTC_START 	= str(UTC_START[0]+'-'+UTC_START[1])
MJD_START       = str((Time(dt)).mjd)
PICOSECONDS 	= str(int((np.array(dt, dtype='datetime64[ns]') - np.array(dt, dtype='datetime64[s]'))*1000))
TSAMP 		= str(1/((np.float64(sample_rate/2))*2))
RESOLUTION 	= str(1)
END             = '# end of header'

HDR_keys = ['HEADER', 'HDR_VERSION', 'HDR_SIZE', 'DADA_VERSION', 'BW', 'FREQ', 'TELESCOPE', 'RECEIVER', 'INSTRUMENT', 'SOURCE', 'RA', 'DEC', 'MODE',  'NBIT', 'NCHAN', 'NDIM', 'NPOL', 'NDAT', 'OBS_OFFSET', 'UTC_START', 'MJD_START', 'PICOSECONDS', 'TSAMP', 'RESOLUTION', 'END']

# open the file in write mode
dada_file = open(out_dada_file, "wb")
logging.warning("out_dada_file: "+ str(out_dada_file))

for key in HDR_keys:
    
    dada_file.write(str.encode(key))
    dada_file.write(str.encode('       \t'))
    dada_file.write(str.encode(globals()[key]))
    dada_file.write(str.encode('\n'))
    

desired_length = int(4096 - dada_file.tell())
dada_file.write(b'\x00' * desired_length)


# --------------------------- HDR section ----------------------------


def raw2ccTseries( file_name, thread_queue, sharedlist, PCAP_Global_HDR=PCAP_Global_HDR, nfft=nfft, udp_pkt_len=udp_pkt_len, HDR_len=HDR_len, CH_sel=CH_sel):
    
    file_size       = os.path.getsize(file_name)
    num_packets     = int((file_size - PCAP_Global_HDR) / udp_pkt_len)
    num_spectrum    = int(num_packets / (nfft / max_nfft_per_pkt))
    num_outerloop   = int(num_spectrum)
    obs_time        = (num_packets * nfft)/33e6

    crnt_file_fid = open(file_name,'rb')
    crnt_file_fid.seek(PCAP_Global_HDR)
    npol_samp       = int( 2 * nfft )
    samples_to_file = np.empty((num_outerloop * nfft * int(NPOL)), dtype = np.int8)

    for loop_idx in range(num_outerloop): 

        cc_Tseries_int	= np.empty( nfft, dtype = np.int8)
        
        raw  		= np.fromfile(crnt_file_fid, dtype='int8', count=udp_pkt_len, sep='')    
        hdr  		= raw[:HDR_len]
        data 		= raw[HDR_len:] 
    
        ch_1 		= data[::2]
        ch_2 		= data[1::2]    

        if CH_sel == "0":
            print("writing full polar ", end='\t')
            Tseries_int	= np.empty( npol_samp, dtype = np.int8)
            Tseries_int[0::2] = ch_1
            Tseries_int[1::2] = ch_2
            samples_to_file[ (npol_samp * loop_idx):( npol_samp * loop_idx)+(npol_samp)] = Tseries_int

        if CH_sel == "1":		   
           print("writing ch_1", end='\t')
           samples_to_file[ (nfft * loop_idx):( nfft * loop_idx)+(nfft)] = ch_1 
           
        if CH_sel == "2":		   
            print("writing ch_2", end='\t')
            samples_to_file[ (nfft * loop_idx):( nfft * loop_idx)+(nfft)] = ch_2    
        
        # Voltage Addition    
        if CH_sel == "3":			
            print("writing ch_1+ch_2", end='\t')
            fft_data_ch1   	= perform_fft(ch_1)                   # No Fliping spectrum
            fft_data_ch2   	= perform_fft(ch_2)    
            
            # Adding Fraction of Phase    
            volt_add    	= V_addition(fft_data_ch1, fft_data_ch2)
        
            # Power level normalization
            scale_facter = np.mean(abs(volt_add)) / ((np.mean(abs(fft_data_ch1)) + np.mean(abs(fft_data_ch2)))/2) 
    
            CC_Tseries 	= perform_ifft(volt_add/scale_facter)				# Tuned to match with the RMS of the signals
            CC_Tseries	= np.round(CC_Tseries)

            cc_Tseries_int = CC_Tseries.real
            #cc_Tseries_int[1::2] = CC_Tseries.imag    
            samples_to_file[ (nfft * loop_idx):( nfft * loop_idx)+(nfft)] = cc_Tseries_int                  
        
        # Cross Correlation   		           	         
        if CH_sel == "4":			
            print("writing ch_1*ch_2", end='\t')
            fft_data_ch1   	= perform_fft(ch_1)                   # No Fliping spectrum
            fft_data_ch2   	= perform_fft(ch_2)    
            
            # Adding Fraction of Phase
            #random_phase 	= np.random.uniform(low=-0.4, high=0.4, size=(nfft,))
            #fft_data_ch2 	= fft_data_ch2 * (1* np.exp(1j*random_phase))
    
            correlation    	= cross_correlation(fft_data_ch1, fft_data_ch2)
        
            # Power level normalization
            scale_facter = np.mean(abs(correlation)) / ((np.mean(abs(fft_data_ch1)) + np.mean(abs(fft_data_ch2)))/2) 
    
            CC_Tseries 	= perform_ifft(correlation/scale_facter)				# Tuned to match with the RMS of the signals
            CC_Tseries	= np.round(CC_Tseries)
            
            cc_Tseries_int = CC_Tseries.real
            #cc_Tseries_int[1::2] = CC_Tseries.imag    
            samples_to_file[ (nfft * loop_idx):( nfft * loop_idx)+(nfft)] = cc_Tseries_int   
    
        print(str((loop_idx/num_outerloop)*100) + '       percentage over', end='\r')
                   
    thread_queue.put(samples_to_file)



def spawn_multiprocessing(files, dada_file):

    for F_num in range(len(files)):
        #manager = mp.Manager()
        #shared_list = manager.list()
        globals()['thread_queue%s' % F_num] = mp.Queue()
        globals()['thread_manager%s' % F_num] = mp.Manager()
        #globals()['thread%s' % F_num] = mp.Process(target=lambda: raw2ccTseries(files[F_num], globals()['thread_queue%s' % F_num], shared_list))
        globals()['thread%s' % F_num] = mp.Process(target=raw2ccTseries, args = (files[F_num], globals()['thread_queue%s' % F_num], globals()['thread_manager%s' % F_num].list()))
        globals()['thread%s' % F_num].start()

    # get processed data from thread
    for F_num in range(len(files)): 
        globals()['thread%s_data' % F_num] = globals()['thread_queue%s' % F_num].get()
        
    for F_num in range(len(files)): 
        globals()['thread%s' % F_num].join()
  
    for F_num in range(len(files)):     
        globals()['thread%s' % F_num].close()
    
    print('\n')
    
    # writing data to file         
    for F_num in range(len(files)):
        globals()['thread%s_data' % F_num].tofile(dada_file)
        print('data '+ str(F_num)+ ' written to '+ str(out_dada_file))  
        
        # deleting variables
        del globals()['thread%s' % F_num]
        del globals()['thread_queue%s' % F_num]         
        del globals()['thread%s_data' % F_num] 
        
    #gc.collect()       


# Master handel for multiprocessing

step_loop = len(files) // no_of_cores

if (step_loop - (len(files) / no_of_cores)) != 0:
    num_of_loop = step_loop + 1
else:
    num_of_loop = step_loop

print('Number of files to process: '+ str(len(files)))
print('Files per batch: '+ str(no_of_cores))
logging.warning("Number of files to process: "+ str(len(files)))
logging.warning("Files per batch:  "+ str(no_of_cores))

for mp_loop_idx in range(num_of_loop):

    print('Batch: '+ str(mp_loop_idx+1)+'/'+ str(num_of_loop))
    if mp_loop_idx == step_loop:
        spawn_multiprocessing(files[int(mp_loop_idx*no_of_cores):], dada_file)
    else:    
         spawn_multiprocessing(files[(mp_loop_idx*no_of_cores):(mp_loop_idx*no_of_cores)+no_of_cores], dada_file)

# closing the file
dada_file.close()

finish_time = datetime.datetime.now()
print(finish_time)
print('Time taken for PDR2DADA convertion: '+ str(finish_time - start_time))
logging.warning("Time taken for PDR2DADA convertion: "+ str(finish_time - start_time))

core_lst = list(np.int8(np.linspace(0,(no_of_cores-1),no_of_cores))) 
cores_to_use = '0,'+",".join([str(item) for item in core_lst if item])

int_time = str(3.0) #str(file_list[4])
#nbins = str(256) #str(128) #str(file_list[5])
#fil_chan = str(str(256)+':1024') #str(str(file_list[6])+':D')
fil_chan = str(256)
#cpu = str('0,1,2,3,4,5')
cpu = str(cores_to_use[4:])
thread = str('32')
nbit = str('8')

if psrname == 'J0534+2200':
    time_avg = str(128e-06)
    nbins = str(128) 
else:
    time_avg = str(256e-06)
    nbins = str(256) 

MJD_START_str = str(np.round(float(MJD_START),2))
output = str(psrname+"_"+MJD_START_str+"_"+Center_freq+"MHz_type_"+str(CH_sel)+"_GBD_DART")

print("frequency, int_time, nbins, fil_chan, thread, nbit, mjd, output ")
print(Center_freq, int_time, nbins, fil_chan, thread, nbit, MJD_START, output)

logging.warning("Center_freq: "+ str(Center_freq))
logging.warning("int_time: "+ str(int_time))
logging.warning("nbins: "+ str(nbins))
logging.warning("fil_chan: "+ str(fil_chan))
logging.warning("cpu: "+ str(cpu))
logging.warning("thread: "+ str(thread))
logging.warning("nbit: "+ str(nbit))
logging.warning("time_avg: "+ str(time_avg))
logging.warning("MJD_START: "+ str(MJD_START))
logging.warning("output: "+ str(output))

singularity_container = 'singularity exec /home/dsp/singularity_imag/pschive_py3.sif  '
logging.warning("singularity_container: "+ str(singularity_container))

input_data = out_dada_file

# ---- Search Mode FITS -------
full_stokes = str(4)
digifits_full_stokes_data = input_data+'_stokes.fits'

#digifits -b 8 -p 4 -F 256:16 -x 2048 -t 128e-06 J0953+0755_2025_05_29_17_33_33_type_0.dada -o test_dada_digifits_p4_x2048.fits

digifits_ful_stokes_cmd = (singularity_container+'digifits ' + ' -F '+fil_chan+' -x ' + fil_chan +' -b ' +nbit+ ' -p ' +full_stokes+ ' -t ' +time_avg+ ' '+ input_data+' -o ' +digifits_full_stokes_data)

# ------ Total intensity FITS -------

total_intensity = str(1)
digifits_total_intensity_data = input_data+'_intensity.fits'

digifits_total_intensity_cmd = (singularity_container+'digifits ' + ' -F '+fil_chan+' -x ' + fil_chan + ' -b ' +nbit+ ' -p ' +total_intensity+ ' -t ' +time_avg+ ' '+ input_data+' -o ' +digifits_total_intensity_data)

# ------ Total intensity FITS -------

# ------- coherent Dedispersion Folded archive -------

dspsr_cmd = (singularity_container+'dspsr '+input_data+' -cpu '+cpu+ ' -t ' + thread +' -E '+parfile+ ' -L '+int_time+ ' -b '+nbins+ ' -F '+fil_chan+':D'+ ' -N '+psrname+ ' -a PSRFITS -A -e .fits -O '+output)
# ------- coherent Dedispersion Folded archive -------

# ------- Incoherent Dedispersion Folded archive -------

Incoherent_DD_out = output+'_stokes_Incoherent_DD'

Incoherent_DD_dspsr_cmd = (singularity_container+'dspsr '+digifits_full_stokes_data+' -cpu '+cpu+ ' -t ' + thread +' -E '+parfile+ ' -L '+int_time+ ' -b '+nbins+ ' -N '+psrname+ ' -A -e .fits -O '+Incoherent_DD_out)

#dspsr J1921+2153_2025_05_02_21_39_52_type_2.dada_10ms.fil -E /home/dsp/parfiles/J1921+2153.par -L 10.0 -b 128 -F 256:D -N J1921+2153 -A -e .fits -O J1921+2153_60797.67352531889_175type_2.gbdnorfix_milt

# ------ Incoherent Dedispersion Folded archive -------

fits_input = '*_GBD_DART*fits'
psredit_cmd = (singularity_container+'psredit -c name='+psrname+',be:name=GSB,coord='+RA_DEC+' -m '+fits_input)

Iterative_cleaner_cmd = (singularity_container+' python3 '+ RFI_mit_pyscript + " -s 3 -c 3 -m 10 " +fits_input)

#pdmp_input = output+'.fits_cleaned.ar'
#pdmp_cmd = (singularity_container+ 'pdmp -g '+output+'.ps/cps '+pdmp_input)
#pdmp_cmd = (singularity_container+ 'pdmp -dr 2 -ds 0.05 -g '+output+'.ps/cps '+pdmp_input)


#DIGIFIL
print('------ DIGIFITS ------')
logging.warning("------ DIGIFITS ------")
digifits_start_time = datetime.datetime.now()
print('---- Search Mode FITS -------')
logging.warning("---- Search Mode FITS -------")
print(str(digifits_ful_stokes_cmd))
logging.warning("digifits_ful_stokes_cmd: "+ str(digifits_ful_stokes_cmd))
os.system(digifits_ful_stokes_cmd)
print('------ Total intensity FITS -------')
logging.warning("------ Total intensity FITS -------")
print(str(digifits_total_intensity_cmd))
logging.warning("digifits_total_intensity_cmd: "+ str(digifits_total_intensity_cmd))
os.system(digifits_total_intensity_cmd)
print('Nchan: '+ fil_chan +', Nbin: '+ nbins+', Tavg: '+ str(time_avg)+' Sceconds')
logging.warning("Nchan: "+ str(fil_chan))
logging.warning("Nbin: "+ str(nbins))
#ogging.warning("Nspec_AVG: "+ str(Nspec_AVG))
logging.warning("Tavg: "+ str(time_avg))
digifits_finish_time = datetime.datetime.now()

print('Time taken for DIGIFITS run: '+ str(digifits_finish_time - digifits_start_time))
logging.warning("Time taken for DIGIFITS run: "+ str(digifits_finish_time - digifits_start_time))

# DSPSR
print('------ DSPSR ------')
logging.warning("------ DSPSR ------")
dspsr_start_time = datetime.datetime.now()
print(' ------- Coherent Dedispersion Folded archive -------')
logging.warning("------- Coherent Dedispersion Folded archive -------")
print(str(dspsr_cmd))
logging.warning("dspsr_cmd: "+ str(dspsr_cmd))
os.system(dspsr_cmd)
print(' ------- Incoherent Dedispersion Folded archive -------')
logging.warning("------- Incoherent Dedispersion Folded archive -------")
print(str(Incoherent_DD_dspsr_cmd))
logging.warning("Incoherent_DD_dspsr_cmd: "+ str(Incoherent_DD_dspsr_cmd))
os.system(Incoherent_DD_dspsr_cmd)
dspsr_finish_time = datetime.datetime.now()

print('Time taken for DSPSR run: '+ str(dspsr_finish_time - dspsr_start_time))
logging.warning("Time taken for DSPSR run: "+ str(dspsr_finish_time - dspsr_start_time))


# PSREDIT
print('------ PSREDIT ------')
logging.warning("------ PSREDIT ------")
psredit_start_time = datetime.datetime.now()
print(str(psredit_cmd))
logging.warning("psredit_cmd: "+ str(psredit_cmd))
os.system(psredit_cmd)
psredit_finish_time = datetime.datetime.now()
print('Time taken for PSREDIT run: '+ str(psredit_finish_time - psredit_start_time))
logging.warning("Time taken for PSREDIT run: "+ str(psredit_finish_time - psredit_start_time))

# Iterative RFI Cleaner
print('------ Iterative RFI Cleaner ------')
logging.warning("------ Iterative RFI Cleaner ------")
RFI_start_time = datetime.datetime.now()
print(str(Iterative_cleaner_cmd))
logging.warning("Iterative_cleaner_cmd: "+ str(Iterative_cleaner_cmd))
os.system(Iterative_cleaner_cmd)
RFI_finish_time = datetime.datetime.now()
print('Time taken for Iterative_cleaner run: '+ str(RFI_finish_time - RFI_start_time))
logging.warning("Time taken for Iterative_cleaner run: "+ str(RFI_finish_time - RFI_start_time))

# ------------------------------
# PDMP
print('------ PDMP ------')
logging.warning("------ PDMP ------")
pdmp_input = glob.glob("*.fits_cleaned.ar")
pdmp_cmd1 = (singularity_container+ 'pdmp -dr 2 -ds 0.05 -g '+pdmp_input[0]+'.ps/cps '+pdmp_input[0])
pdmp_cmd2 = (singularity_container+ 'pdmp -dr 2 -ds 0.05 -g '+pdmp_input[1]+'.ps/cps '+pdmp_input[1])

pdmp_start_time = datetime.datetime.now()
print(str(pdmp_cmd1))
logging.warning("pdmp_cmd1: "+ str(pdmp_cmd1))
os.system(pdmp_cmd1)
print(str(pdmp_cmd2))
logging.warning("pdmp_cmd2: "+ str(pdmp_cmd2))
os.system(pdmp_cmd2)
pdmp_finish_time = datetime.datetime.now()
print('Time taken for PDMP run: '+ str(pdmp_finish_time - pdmp_start_time))
logging.warning("Time taken for PDMP run: "+ str(pdmp_finish_time - pdmp_start_time))
# ------------------------------

# PS2PDF
print('------ PS2PDF ------')
logging.warning("------ PS2PDF ------")
#ps2pdf_input = (output+'.ps')
ps2pdf_input = glob.glob("*.ps")
#ps2pdf_cmd = (singularity_container+'ps2pdf '+ ps2pdf_input+' '+ output +'.pdf')
ps2pdf_cmd1 = (singularity_container+'ps2pdf '+ ps2pdf_input[0]+' '+ ps2pdf_input[0] +'.pdf')
ps2pdf_cmd2 = (singularity_container+'ps2pdf '+ ps2pdf_input[1]+' '+ ps2pdf_input[1] +'.pdf')

ps2pdf_start_time = datetime.datetime.now()
print(ps2pdf_cmd1)
logging.warning("ps2pdf_cmd1: "+ str(ps2pdf_cmd1))
os.system(ps2pdf_cmd1)
print(ps2pdf_cmd2)
logging.warning("ps2pdf_cmd2: "+ str(ps2pdf_cmd2))
os.system(ps2pdf_cmd2)
ps2pdf_finish_time = datetime.datetime.now()
print('Time taken for PDMP run: '+ str(ps2pdf_finish_time - ps2pdf_start_time))
logging.warning("Time taken for PDMP run: "+ str(ps2pdf_finish_time - ps2pdf_start_time))

print('Overall time taken for data reduction: '+ str(ps2pdf_finish_time - start_time))
logging.warning("Overall time taken for data reduction: "+ str(ps2pdf_finish_time - start_time))

rm_cmd = 'rm ' + out_dada_file
os.system(rm_cmd)

print(out_dada_file + '... file removed')
logging.warning(out_dada_file + '... file removed')

print('All done!')
logging.warning("All done!")

"""

singularity_container = 'singularity exec /home/inpta/pschive_py3.sif '

singularity exec /home/inpta/pschive_py3.sif dspsr unset.dada -cpu 0,1,2,3,4,5,6,7,8,9,10,11 -t 12 -F 256[:D] -c 1.0 -D 0.0 -L 2.0 -e .fits -AO unset

singularity shell ~/singularity_imag/pschive_py3.sif

dspsr unset.dada -cpu 0,1,2,3,4,5,6,7,8,9 -t 10 -F 256[:D] -c 1.0 -D 0.0 -L 2.0 -e .fits -AO unset

psrplot -p freq+ -jDT unset.fits -D /xs

psrplot -p time -jDF unset.fits -D /xs

pdmp -dr 1 -ds 10 -g unset.ps/cps unset.fits 

pdmp -dr 1 -ds 1 -g unset.fits.pazi.ps/cps unset.fits.pazi 

"""     
