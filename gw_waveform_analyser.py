#!/usr/bin/env python
# coding: utf-8

# In[20]:


import h5py 
import numpy as np
import matplotlib.pyplot as plt 
import cmath
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy import signal


# In[3]:


#Setting file locations 
simdir = []
for i in range(1,17):
    runnum = 'R01'
    if(i<10):
        simname = 'THC_000' + str(i)
        simdir.append('/home/sanikawork/Desktop/Sathya/CoRe/' + simname + '/' + runnum + '/data.h5')
    else:
        simname = 'THC_00' + str(i)
        simdir.append('/home/sanikawork/Desktop/Sathya/CoRe/' + simname + '/' + runnum + '/data.h5')

#Reading the file 
data = []
for i in range(16):
    file = h5py.File(simdir[i],'r')
    data.append(np.array(file['rh_22']['Rh_l2_m2_r00400.txt'])) 
    
#Creating data dictionary 
data_keys = {'1':'1.25-1.25','2':'1.3-1.3','3':'1.35-1.35','4':'1.365-1.25','5':'1.4-1.2',
             '6':'1.4-1.4','7':'1.44-1.39','8':'1.5-1.5','9':'1.6-1.6'}


# In[4]:


fig ,axs = plt.subplots(4,4,figsize=(15,9))
temp = 0 
for i in range(4):
    for j in range(4):
        realmax = np.argmax(data[temp][:,1])
        imgmax = np.argmax(data[temp][:,2])
        data[temp][:realmax,1] = 0
        data[temp][:imgmax,2] = 0
        realh = data[temp][:,1]
        imgh = data[temp][:,2]
        merge_time = data[temp][:,8]*5e-6
        axs[i,j].plot(merge_time, realh)
        temp = temp + 1 
fig.suptitle('All THC waveforms',fontsize=15)


# In[5]:


fig ,axs = plt.subplots(3,3,figsize=(17,10))
#fig.tight_layout()
temp = 0 
for i in range(3):
    for j in range(3):
        realmax = np.argmax(data[temp][:,1])
        imgmax = np.argmax(data[temp][:,2])
        data[temp][:realmax,1] = 0
        data[temp][:imgmax,2] = 0
        realh = data[temp][:,1]
        imgh = data[temp][:,2]
        merge_time = data[temp][:,8]*5e-6
        axs[i,j].plot(merge_time, realh)
        temp = temp + 1 
        axs[i,j].set_title('THC000'+str(temp)+' | Masses '+data_keys[str(temp)])
fig.suptitle('BHBlp Waveforms',fontsize=15)
fig.savefig('BHBlp_all.pdf')


# In[38]:


fig ,axs = plt.subplots(3,3,figsize=(17,10))
temp = 0 
fpeaks = []
for i in range(3):
    for j in range(3):
        realh = data[temp][:,1]
        imgh = data[temp][:,2]
        signal_gw = realh + 1j*imgh
        sigmax = np.argmax(signal_gw)
        signal_gw[:sigmax] = 0
        N = data[temp][:,0].shape[0]
        merge_time = data[temp][:,8]*5e-6
        fs = N/merge_time[-1]
        T= (merge_time[1]-merge_time[0])*5e-6
        fsignal = fft(signal_gw)
        freqs = fftfreq(N,1/fs)
        temp = temp + 1 
        term = np.sqrt(freqs[0:N//2])*np.abs(fsignal[0:N//2]) #characteristic amp multiplication 
        #term = np.abs(fsignal[0:N//2])
        fpeaks.append(freqs[np.argmax(term)]/1000)
        print(freqs[np.argmax(term)])
        axs[i,j].set_title('THC000'+str(temp)+' | Masses '+data_keys[str(temp)])
        axs[i,j].plot(freqs[0:N//2],term)
        axs[i,j].set_xlim(0,5000)
fig.suptitle('Power Spectral density for BHBlp Waveforms',fontsize=15)
fig.savefig('PSD_bhblp.pdf')


# In[19]:


def r_16(fpeak):
    return (6.284 - fpeak)/0.2823
fpeak = [0,4]
r_16_emp = []
r_16_core = []
print(fpeaks)
for i in range(len(fpeak)):
    r_16_emp.append(r_16(fpeak[i])) 
for i in range(len(fpeaks)):
    r_16_core.append(r_16(fpeaks[i]))
plt.figure(figsize=(12,7))
plt.plot(r_16_emp[:3],fpeak[:3],color='black',label='Emprirical fit Bauswein Et al.')
#plt.plot(r_16_core[0],fpeaks[0],'*',label='CoRe Database Equal Mass 2.5')
#plt.plot(r_16_core[1],fpeaks[1],'*',label='CoRe Database Equal Mass 2.7')
#plt.plot(r_16_core[2],fpeaks[2],'*',label='CoRe Database Equal Mass 2.6')
plt.plot(r_16_core[:3],fpeaks[:3],'*',color='red',label='CoRe Database')
for i in range(3):
    plt.annotate(data_keys[str(i+1)], (r_16_core[i], fpeaks[i]))
plt.axvline(13.15,label='$R_{1.6}$ for $BHB\Lambda\Phi$')
plt.xlabel('$R_{1.6}$',fontsize=15)
plt.ylabel('$f_{peak}$',fontsize=15)
plt.legend(fontsize=15)
plt.savefig('all_bhblp_r_1.6.pdf')


# In[15]:


def r_16(fpeak):
    return (6.284 - fpeak)/0.2823
fpeak = [0,4]
r_16_emp = []
r_16_core = []
print(fpeaks)
for i in range(len(fpeak)):
    r_16_emp.append(r_16(fpeak[i])) 
for i in range(len(fpeaks)):
    r_16_core.append(r_16(fpeaks[i]))
plt.figure(figsize=(12,7))
plt.plot(r_16_emp,fpeak,color='black',label='Emprirical fit Bauswein Et al.')
plt.plot(r_16_core[0],fpeaks[0],'*',label='CoRe Database Equal Mass 2.5')
plt.plot(r_16_core[1],fpeaks[1],'*',label='CoRe Database Equal Mass 2.6')
plt.plot(r_16_core[2],fpeaks[2],'*',label='CoRe Database Equal Mass 2.7')
#plt.plot(r_16_core,fpeaks,'*',color='red',label='CoRe Database')
for i in range(3):
    plt.annotate(data_keys[str(i+1)], (r_16_core[i], fpeaks[i]))
plt.axvline(13.25,label='$R_{1.6}$ for $BHB\Lambda\Phi$')
plt.xlabel('$R_{1.6}$',fontsize=15)
plt.ylabel('$f_{peak}$',fontsize=15)
plt.legend(fontsize=15)
plt.savefig('imp_bhblp_r16.pdf')


# In[ ]:




