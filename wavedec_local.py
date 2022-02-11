#Dated: Jan 2022
#Author: Sanika S. Khadkikar
#Pennsylvania State University

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import h5py
import argparse
from watpy.coredb.coredb import *

#Global constants
time_con_f = 4.975e-6

# Creating the parser
parser = argparse.ArgumentParser()

# Adding arguments
parser.add_argument('--dpath', type=str, required=True, help = 'Path for the CoRe simulation directory for local implementation')
parser.add_argument('--runno', type=str, required=False, help = 'Run to pick from the dpath directory. Default is --runno=R01', default='R01')
parser.add_argument('--exctrad', type=float, required=False, help = 'Extraction radius at which the gravitational waveform was picked in solar masses', default=400)
parser.add_argument('--gwplot', type=bool, required=False, help = 'Plot the gravitational wave being analysed as a sanity check and save the image', default=False)
parser.add_argument('--scale_min', type=float, required=False, help='Minimum value to be used for the scale array to be fed to the Continuous Wavelet Transform', default=1)
parser.add_argument('--scale_max', type=float, required=False, help='Maximum value to be used for the scale array to be fed to the Continuous Wavelet Transform', default=150)
parser.add_argument('--dscale', type=float, required=False, help='Spacing between the scales', default=1)


# Parse these arguments
args = parser.parse_args()

#Loading in the THC_0001 CoRe Waveform
fpath = args.dpath + '/' + args.runno + '/data.h5'
sub_key = 'Rh_l2_m2_r00' + str(args.exctrad) + '.txt'
data = h5py.File(fpath, 'r')

#Defining the gravitational waveform(gwf)
gwf = np.array(data['rh_22'][sub_key])
hplus = gwf[:,1]
hcross = -gwf[:,2]
time = gwf[:,8]*time_con_f                         #converting to milliseconds
env = np.sqrt(hplus**2 + hcross**2)
n = len(hplus)

#Cutting inspiral off
env_max = np.argmax(env)

for i in range(env_max, n):
    if env[i] < env[i+1]:
        cut_point = i
        break
        
postmerger = hplus[cut_point:]
pm_time = time[cut_point:]     


#Plotting the gwf
if args.gwplot:
	fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
	ax[0].plot(time, hplus, label = r'$h_+$ - Real Strain')
	ax[0].plot(time, env, label = 'Magnitude of Strain')
	ax[0].legend()
	ax[1].plot(pm_time, postmerger)
	ax[0].set_xlabel('Time (ms)', fontsize=10)
	ax[0].set_ylabel('Strain')
	ax[0].set_title('Binary neutron star merger signal under consideration')
	ax[1].set_xlabel('Time (ms)')
	ax[1].set_ylabel('Strain')
	ax[1].set_title('Binary neutron star postmerger signal under consideration')
	plt.savefig('gwfplot.pdf', fontsize=10)
	plt.show()


#Defining sampling period and frequency
sam_p = (pm_time[-1] - pm_time[0])/len(pm_time)
sam_f = 1/sam_p

#Defining scale for the wavelet analysis
scales = np.arange(args.scale_min, args.scale_max, args.dscale)

#CWT on the gwf using the Morlet wavelet
coefs, freqs = pywt.cwt(postmerger, scales, 'morl', sampling_period = sam_p)

#Normalising the coefficient matrix using the Frobenius norm
norm_mat = (np.abs(coefs))/(np.linalg.norm(coefs))

#Reconstructing the signal
reconstructed = np.zeros_like(postmerger)
for n in range(len(pm_time)):
	reconstructed[n] = np.sum(coefs[:,n]/scales**0.5)
#reconstructed = reconstructed/np.max(reconstructed)*np.max(postmerger)
print(np.max(reconstructed)/np.max(postmerger))
print(n)
#Plotting the wavelet transform coefficients
X = pm_time
Y = scales
X, Y = np.meshgrid(X, Y)
Z = norm_mat
fig,ax=plt.subplots(2,1, figsize=[10,10])
cp = ax[0].contourf(X, Y, Z, cmap = cm.inferno)
fig.colorbar(cp, ax=ax[0]) # Add a colorbar to a plot
ax[0].set_title('Continuous Wavelet transform - Contour Plot')
ax[0].set_xlabel('Time (code units)')
ax[0].set_ylabel('Scale')
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('Strain')
ax[1].set_title('Post reconstruction comparison')
ax[1].plot(pm_time, postmerger, label='Original Signal')
ax[1].plot(pm_time, reconstructed,'.', label='Reconstructed Signal')
ax[1].legend()
plt.savefig('gwave_trans.pdf', bbox_inches="tight")
#plt.show()
