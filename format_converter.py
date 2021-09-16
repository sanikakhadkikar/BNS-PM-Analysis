""" This is a code that converts Computational Relativity database
 waveforms into the LIGO standard NRHDF5 format"""

#Importing libraries
import h5py
import numpy as np
import romspline


#File selection
file_name = '/home/sanikawork/Desktop/Sathya/CoRe/THC_0001/R01/data.h5'
gw_core = h5py.File(file_name, 'r')
keys = list(gw_core['rh_22'].keys())
gw_core['rh_22'].keys()


#Required inputs
file_name = 'THC_0001_conv.h5'
l_max = 4

df = h5py.File(file_name, 'a')
df.attrs['EOS-name'] = 'SLy'
df.attrs['EOS-references'] = 'DePietri_2015'
df.attrs['EOS-remarks'] = ''
df.attrs['Format'] = 1
df.attrs['INSPIRE-bibtex-keys'] = 'DePietri_2015'
df.attrs['LNhatx'] = 0
df.attrs['LNhaty'] = 0
df.attrs['LNhatz'] = 1
df.attrs['NR-code'] = 'ET+GRHydro'
df.attrs['NR-group'] = 'CoRe'
df.attrs['NR-techniques'] = 'Quasi-Circular-Irrot-ID,BSSN,HLLE,WENO5,Psi4-integrate, Extrapolated-Waveform'
df.attrs['NS-spins-meaningful'] = True
df.attrs['Omega'] = 0.02119356623877212
df.attrs['alternative-names'] = 'SLy_1.20vs1.20_d40.0km_15.1ms'
df.attrs['baryonic-mass1-msol'] = 1.2
df.attrs['baryonic-mass2-msol'] = 1.2
df.attrs['bns-remnant-collapsed'] = False
df.attrs['eccentricity'] = 0.0
df.attrs['eta'] = 0.25
df.attrs['f_lower_at_1MSUN'] = 1369.6343414914013
df.attrs['file-format-version'] = 2
df.attrs['files-in-error-series'] = ''
df.attrs['have-ns-tidal-lambda'] = False
df.attrs['license'] = 'LVC-internal'
df.attrs['mass1'] = 0.5
df.attrs['mass1-msol'] = 1.113579578904359
df.attrs['mass2'] = 0.5
df.attrs['mass2-msol'] = 1.113579578904359
df.attrs['mean_anomaly'] = -1.0
df.attrs['modification-date'] = '09-16-2021'
df.attrs['name'] = 'SLy_Mtot_2227_q_100_dx_369_D40_DePietri_2015lya'
df.attrs['nhatx'] = 1
df.attrs['nhaty'] = 0
df.attrs['nhatz'] = 0
df.attrs['object1'] = 'NS'
df.attrs['object2'] = 'NS'
df.attrs['point-of-contact-email'] = 'sanikas.khadkikar@ligo.org'
df.attrs['production-run'] = 0
df.attrs['simulation-type'] = 'non-spinning'
df.attrs['spin1x'] = 0
df.attrs['spin1y'] = 0
df.attrs['spin1z'] = 0
df.attrs['spin2x'] = 0
df.attrs['spin2y'] = 0
df.attrs['spin2z'] = 0
df.attrs['type'] = 'NRinjection'



for l in range(l_max+1):
  for m in range(0, l+1, 1):
    count = 0
    temp_r = np.array(gw_core['rh_22'][keys[count]][:, 1])
    temp_i = np.array(gw_core['rh_22'][keys[count]][:, 2])
    comp = temp_r + 1j*temp_i
    mag = np.abs(comp)
    df = h5py.File(file_name, 'a')
    str_name = 'amp_l'+str(l)+'_m'+ str(m)
    df.create_group(str_name)
    time_ar = np.array(gw_core['rh_22'][keys[count]][:, 8])
    spline = romspline.ReducedOrderSpline(time_ar, mag, verbose=False)
    df[str_name].create_dataset('X', data=spline.X)
    df[str_name].create_dataset('Y', data=spline.Y)
    df[str_name].create_dataset('deg', data=[5])
    df[str_name].create_dataset('errors', data=spline.errors)
    df[str_name].create_dataset('tol', data=spline.tol)
    df.close()
    print(str_name + ' is done')


df = h5py.File(file_name, 'a')
df.create_group('auxillary info')
df.close()

for l in range(l_max+1):
  for m in range(0, l+1, 1):
    count = 0
    temp_r = np.array(gw_core['rh_22'][keys[count]][:, 1])
    temp_i = np.array(gw_core['rh_22'][keys[count]][:, 2])
    comp = temp_r + 1j*temp_i
    phase = np.angle(comp)
    df = h5py.File(file_name, 'a')
    str_name = 'phase_l'+str(l)+'_m'+ str(m)
    df.create_group(str_name)
    time_ar = np.array(gw_core['rh_22'][keys[count]][:, 8])
    spline = romspline.ReducedOrderSpline(time_ar, phase, verbose=False)
    df[str_name].create_dataset('X', data=spline.X)
    df[str_name].create_dataset('Y', data=spline.Y)
    df[str_name].create_dataset('deg', data=[5])
    df[str_name].create_dataset('errors', data=spline.errors)
    df[str_name].create_dataset('tol', data=spline.tol)
    df.close()
    print(str_name + ' is done')


new_file = h5py.File(file_name, 'r')
new_file.keys()

print(keys)

new_file.close()
