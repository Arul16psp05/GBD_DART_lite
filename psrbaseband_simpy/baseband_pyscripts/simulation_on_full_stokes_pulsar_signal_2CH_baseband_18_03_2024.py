"""
# Read full stokes data
# Reference https://github.com/supremeKAI40/PulsarDataPlotting/blob/main/Pulsar_radio_polarisation_data.ipynb

by Arul @ 17_03_2024

Not finished

"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from baseband_tasks.generators import NoiseGenerator
from astropy import time as T, units as u
from scipy import signal
from datetime import timezone 
import datetime 
#import psrsigsim as pss
   
filename = '/home/arul/data2/python_polarization/pulsar_data/B1929+10/B1929+10-410MHz_stc99_410.fits'

psrname = "B1929+10"


hdul = fits.open(filename)
hdul.info()


params = hdul[0]
subint = hdul[2]     # Change accordingly

params.header
subint.header
subint.columns

# Getdata

data = subint.data[0][-1]
data.shape

# Stokes data extraction

I = data[0]
Q = data[1]
U = data[2] 
V = data[3]
print(I.shape, Q.shape, U.shape, V.shape)


#DAT Offset

dataOFF = subint.data['DAT_OFFS']
dataOFF

#Data Scale Factor
dataSCL= subint.data['DAT_SCL']
dataSCL

# Normalization

def scaling(dataArr, dataSCL, dataOFF):
    rawDATAi= (dataSCL[0,0]*dataArr[0]) + dataOFF[0,0]
    rawDATAq= (dataSCL[0,1]*dataArr[1]) + dataOFF[0,1]
    rawDATAu= (dataSCL[0,2]*dataArr[2]) + dataOFF[0,2]
    rawDATAv= (dataSCL[0,3]*dataArr[3]) + dataOFF[0,3]
    return rawDATAi, rawDATAq, rawDATAu, rawDATAv
    
# Different Channel Data Isolation

DATA_I, DATA_Q, DATA_U, DATA_V = scaling(data, dataSCL, dataOFF)



# Function to determine baseline average of a I, Q, U, or V array

def baselineSUB(unscaledData):
    sortedData   = np.sort(unscaledData)
    baselineData = sortedData[0,300:612] # sortedData[0,:] # 
    return unscaledData-np.mean(baselineData)


# Code to convert raw IQUV data to IQUV data of the pulsar

DATA_I_Corrected = baselineSUB(DATA_I)
DATA_Q_Corrected = baselineSUB(DATA_Q)
DATA_U_Corrected = baselineSUB(DATA_U)
DATA_V_Corrected = baselineSUB(DATA_V)


PA = np.degrees(0.5* np.arctan((DATA_U_Corrected/DATA_Q_Corrected)))

I = DATA_I_Corrected[0]
Q = DATA_Q_Corrected[0]
U = DATA_U_Corrected[0]
V = DATA_V_Corrected[0]

"""
# Plot

plt.plot(I)
plt.plot(Q)
plt.plot(U)
plt.plot(V)

plt.legend(['I','Q','U','V'])

plt.title(filename.rsplit('/')[-1])

plt.show()
"""

# ------------------------- Baseband section --------------------------
# simulation

out_data_file = psrname+'.dat'
out_hdr_file = psrname+'.hdr'


#Nbin			= len(I)
#Nchan			= 64
sample_rate     	= 33 			# MSps  --> 33 MHz
samples_per_frame 	= int(sample_rate * 1e6)     # 1 sec freame
start_time      	= T.Time.now()
obs_time        	= 10  			# In seconds 
Num_of_sample   	= int(obs_time * samples_per_frame)
sample_std		= 1			# 68 percentage of full span
eight_bit_int_range	= 2**8     
sample_range 		= (((eight_bit_int_range/100)*68)/2)
single_pulse_SNR 	= 1.5          		# for Noise
num_samples_shift	= 1

#Tseries = NoiseGenerator((Num_of_sample,), start_time,sample_rate*u.MHz, samples_per_frame=samples_per_frame, dtype='f2')


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html

# -------------------------Interpolation resampling -------------------

cc_nfft			= 2**10#2**16
period 		    	= 0.2265187466568
rot_per_sec 	    	= 1/period

samples_per_rot     	= round((sample_rate*1e6) / rot_per_sec)

#num_sample_spec 	= samples_per_rot // cc_nfft
#temp_sample		= samples_per_rot - num_sample_spec*(cc_nfft)

I = signal.resample(I, samples_per_rot)
Q = signal.resample(Q, samples_per_rot)
U = signal.resample(U, samples_per_rot)
V = signal.resample(V, samples_per_rot)

# ---------------------------- Interpolation resampling


# Correlation products
# Auto

Ex_Ex = 1/2*(I+Q)  # np.sqrt(Ex_Ex) = abs(X)

Ey_Ey = 1/2*(I-Q)  # np.sqrt(Ey_Ey) = abs(Y)

# Cross correlation

Ex_Ey = 1/2* np.vectorize(complex)(U,+V) # np.sqrt(2*Ex_Ey*np.cos(phi))

Ey_Ex = 1/2* np.vectorize(complex)(U,-V) # np.sqrt(2*Ey_Ex*np.sin(phi))


#D_l = np.sqrt( Q**2 + U**2 ) / I	# Degree of linear polarization

#D_c = V / I 				# Degree of circular polarization

#D_t = np.sqrt( Q**2 + U**2 + V**2) / I	# Degree of linear polarization

#xi = 1/2*(np.arctan(U / Q))		# Position angle

#ratio = abs(Ey_Ey) / abs(Ex_Ex)


def Polar2Cartesian(radii, angles):
    return radii * np.exp(1j*angles)

def Cartesian2Polar(complex_number):
    return np.abs(complex_number), np.angle(complex_number)
    

template_Nbin = len(I)


amp_x_signal  = np.zeros(template_Nbin, dtype=np.float64)
amp_y_signal  = np.zeros(template_Nbin, dtype=np.float64)



for template_sample_idx in range(template_Nbin):
    
    print('Making Voltage Tseries from Stokes: ' + str((template_sample_idx/template_Nbin)*100) + ' pecentage.', end='\r')
    if Ex_Ex[template_sample_idx] < 0:
        amp_x_signal[template_sample_idx] = -(np.emath.sqrt(Ex_Ex[template_sample_idx])).imag
    else:
        amp_x_signal[template_sample_idx] = np.sqrt(Ex_Ex[template_sample_idx])
        
    if Ey_Ey[template_sample_idx] < 0:
        amp_y_signal[template_sample_idx] = -(np.emath.sqrt(Ey_Ey[template_sample_idx])).imag
    else:
        amp_y_signal[template_sample_idx] = np.sqrt(Ey_Ey[template_sample_idx])

 
#np.emath.sqrt(-0.00082746) *1j

#amp_x_signal 		= np.nan_to_num(np.sqrt(Ex_Ex))
#amp_y_signal 		= np.nan_to_num(np.sqrt(Ey_Ey))

cc_amp, cc_phase 	= Cartesian2Polar(Ex_Ey)#cmath.polar(Ex_Ey)[0], cmath.polar(Ex_Ey)[1]
delta_phase		= cc_phase

random_phase = np.random.uniform(low=-np.pi, high=np.pi, size=(template_Nbin,))

phase_x_signal		= random_phase
phase_y_signal		= random_phase - delta_phase

phase_y_signal = (phase_y_signal + np.pi) % (2 * np.pi) - np.pi

delta_phase_2 = phase_x_signal - phase_y_signal
delta_phase_2 = (delta_phase_2 + np.pi) % (2 * np.pi) - np.pi

#plt.plot(np.rad2deg(phase_x_signal), '.')
#plt.plot(np.rad2deg(phase_y_signal), '.')
#plt.show()


x_signal		= Polar2Cartesian(amp_x_signal, phase_x_signal)	
y_signal		= Polar2Cartesian(amp_y_signal, phase_y_signal)


# simulation

Tseries_x = NoiseGenerator((Num_of_sample,), start_time,sample_rate*u.MHz, samples_per_frame=samples_per_frame, dtype=np.complex64)

Tseries_y = NoiseGenerator((Num_of_sample,), start_time,sample_rate*u.MHz, samples_per_frame=samples_per_frame, dtype=np.complex64)

fw = open(out_data_file, "w+")

sample_scale = 8 # Temp. fixed

master_loop =  int(Num_of_sample // samples_per_rot)
for loop_idx in range(master_loop):
    
    Noise_x		= Tseries_x.read(samples_per_rot)
    Noise_y		= Tseries_y.read(samples_per_rot)
               
    x_raw_data 		= ((x_signal * Noise_x) + single_pulse_SNR * Noise_x ) * sample_scale #sample_range 
    y_raw_data 		= ((y_signal * Noise_y) + single_pulse_SNR * Noise_y ) * sample_scale #sample_range

    Tseries_int	= np.empty(4*samples_per_rot, dtype=np.int8)
    #Tseries_int		= np.empty(2*samples_per_rot, dtype=np.int8)
    
    """
    Tseries_int[0::2] = x_raw_data.real
    Tseries_int[1::2] = y_raw_data.real
    """
    Tseries_int[0::4] = x_raw_data.real
    Tseries_int[1::4] = x_raw_data.imag
    Tseries_int[2::4] = y_raw_data.real
    Tseries_int[3::4] = y_raw_data.imag
    
    Tseries_int.tofile(fw)
    
    print( 'Writing to data file: '+ str((loop_idx/master_loop)*100) + ' percentage over', end='\r')

  
fw.close()

# -------------------------------------------------------- HDR section ----------------------------------------------------------

Center_freq     = 200
  
# Getting the current date 
# and time 
dt = np.datetime64(datetime.datetime.now(timezone.utc))


HDR_VERSION 	= '1.0'
BW    		= np.float64(sample_rate/2)
FREQ  		= float(Center_freq)
TELESCOPE 	= 'GMRT'
RECEIVER 	= 'Fake'
INSTRUMENT 	= 'Fake'
# description of the instrument
#TELESCOPE    	= 'Effelsberg'       # telescope name 
#INSTRUMENT   	= 'psrix'           # instrument name
#RECEIVER     	= 'P200-3'          # Frontend receiver
SOURCE  	= psrname
MODE  		= 'PSR'
NBIT 		= int(8)
NCHAN 		= 1
NDIM  		= 2 # real-1, complex-2
NPOL 		= 2
NDAT 		= int(master_loop * samples_per_rot)
BASIS        	= 'Linear'
OBS_OFFSET 	= 0
UTC_START  	= str(dt).rsplit('T')
UTC_START 	= str(UTC_START[0]+'-'+UTC_START[1])
PICOSECONDS 	= int((np.array(dt, dtype='datetime64[ns]') - np.array(dt, dtype='datetime64[s]'))*1000)
TSAMP 		= float(1/(BW*2))
RESOLUTION 	= 2

HDR_keys = ['HDR_VERSION', 'BW', 'FREQ', 'TELESCOPE', 'RECEIVER', 'INSTRUMENT', 'SOURCE', 'MODE', 'NBIT', 'NCHAN', 'NDIM', 'NPOL', 'NDAT', 'BASIS', 'OBS_OFFSET', 'UTC_START', 'PICOSECONDS', 'TSAMP', 'RESOLUTION']

# open the file in write mode
obj_file = open(out_hdr_file, "w")

for key in HDR_keys:
    print(key, "       \t", globals()[key], file=obj_file)

# closing the file
obj_file.close()

# -------------------------------------------------------- HDR section ---------------------------------------------------------

"""
# Simulated stokes

sim1_I = (x_signal*np.conj(x_signal)) + (y_signal*np.conj(y_signal)) 	# I = (Ex_Ex + Ey_Ey)

sim1_Q = (x_signal*np.conj(x_signal)) - (y_signal*np.conj(y_signal))	# Q = (Ex_Ex - Ey_Ey)

sim1_U = (x_signal*np.conj(y_signal)) + (y_signal*np.conj(x_signal)) 	# U = (Ex_Ey + Ey_Ex)

sim1_V = (x_signal*np.conj(y_signal)) - (y_signal*np.conj(x_signal)) 	# V = (Ex_Ey - Ey_Ex)

sim1_I = sim1_I.real
sim1_Q = sim1_Q.real
sim1_U = sim1_U.real
sim1_V = sim1_V.imag

# ----------------------------- 

plt.plot(I, '-k')
plt.plot(Q)
plt.plot(U)
plt.plot(V)

plt.plot(sim1_I, '.-')
plt.plot(sim1_Q, '--')
plt.plot(sim1_U, '--')
plt.plot(sim1_V, '--')

plt.legend(['I','Q','U','V','Sim_I','Sim_Q','Sim_U','Sim_V'])

plt.show()

plt.plot(((I - min(I))/max(I)) - (sim1_I - min(sim1_I))/max(sim1_I))
plt.plot(((Q - min(Q))/max(Q)) - (sim1_Q - min(sim1_Q))/max(sim1_Q))
plt.plot(((U - min(U))/max(U)) - (sim1_U - min(sim1_U))/max(sim1_U))
plt.plot(((V - min(V))/max(V)) - (sim1_V - min(sim1_V))/max(sim1_V))
plt.show()
"""
