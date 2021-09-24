""" This is a code that converts Computational Relativity database
 waveforms into the LIGO standard NRHDF5 format"""

#Importing libraries
import h5py
import numpy as np
import romspline


#File selection
og_file_name = 'data.h5'
gw_core = h5py.File(og_file_name, 'r')
keys = list(gw_core['rh_22'].keys())
gw_core['rh_22'].keys()


#Required inputs
file_name = 'THC_0001_conv.h5'
l_max = 4

df = h5py.File(file_name, 'a')
df.attrs['EOS-name'] = 'BHBlp'
df.attrs['EOS-references'] = 'DePietri_2015'
df.attrs['EOS-remarks'] = ''
df.attrs['Format'] = 1
df.attrs['INSPIRE-bibtex-keys'] = 'DePietri_2015'
df.attrs['LNhatx'] = 0.0
df.attrs['LNhaty'] = 0.0
df.attrs['LNhatz'] = 1.0
df.attrs['Lmax'] = 4
df.attrs['NR-group'] = 'CoRe'
df.attrs['NS-spins-meaningful'] = True
df.attrs['Omega'] = 0.0498819129216
df.attrs['baryonic-mass1-msol'] = 1.35299
df.attrs['baryonic-mass2-msol'] = 1.35299
df.attrs['bns-remnant-collapsed'] = False
df.attrs['eccentricity'] = 0.0
df.attrs['eta'] = 0.25
df.attrs['f_lower_at_1MSUN'] = 644.684
df.attrs['file-format-version'] = 2
df.attrs['files-in-error-series'] = ''
df.attrs['have-ns-tidal-lambda'] = False
df.attrs['license'] = 'LVC-internal'
df.attrs['mass1'] = 0.5
df.attrs['mass1-msol'] = 1.25
df.attrs['mass2'] = 0.5
df.attrs['mass2-msol'] = 1.25
df.attrs['mean_anomaly'] = -1.0
df.attrs['modification-date'] = '09-20-2021'
df.attrs['name'] = 'THC_0001'
df.attrs['nhatx'] = 1.0
df.attrs['nhaty'] = 0.0
df.attrs['nhatz'] = 0.0
df.attrs['object1'] = 'NS'
df.attrs['object2'] = 'NS'
df.attrs['point-of-contact-email'] = 'sanikas.khadkikar@ligo.org'
df.attrs['production-run'] = 0
df.attrs['simulation-type'] = 'non-spinning'
df.attrs['spin1x'] = 0.0
df.attrs['spin1y'] = 0.0
df.attrs['spin1z'] = 0.0
df.attrs['spin2x'] = 0.0
df.attrs['spin2y'] = 0.0
df.attrs['spin2z'] = 0.0
df.attrs['type'] = 'NRinjection'
count = 3

for l in range(2, l_max+1):
    for m in range(0, l+1, 1):
        if m==0:
            temp_r = np.array(gw_core['rh_22'][keys[count]][:, 1])
            temp_i = np.array(gw_core['rh_22'][keys[count]][:, 2])
            comp = temp_r + 1j*temp_i
            mag = np.abs(comp)
            time_ar = np.array(gw_core['rh_22'][keys[count]][:, 8])
            df = h5py.File(file_name, 'a')
            str_name = 'amp_l'+str(l)+'_m'+ str(m)
            print(str_name, count)
            df.create_group(str_name)
            spline = romspline.ReducedOrderSpline(time_ar, mag, verbose=False)
            df[str_name].create_dataset('X', data=spline.X)
            df[str_name].create_dataset('Y', data=spline.Y)
            df[str_name].create_dataset('deg', data=[5])
            df[str_name].create_dataset('errors', data=spline.errors)
            df[str_name].create_dataset('tol', data=spline.tol)
            count = count + 1 
        else:
            temp_r = np.array(gw_core['rh_22'][keys[count]][:, 1])
            temp_i = np.array(gw_core['rh_22'][keys[count]][:, 2])
            comp = temp_r + 1j*temp_i
            mag = np.abs(comp)
            time_ar = np.array(gw_core['rh_22'][keys[count]][:, 8])
            df = h5py.File(file_name, 'a')
            str_name = 'amp_l'+str(l)+'_m'+ str(m)
            df.create_group(str_name)
            spline = romspline.ReducedOrderSpline(time_ar, mag, verbose=False)
            df[str_name].create_dataset('X', data=spline.X)
            df[str_name].create_dataset('Y', data=spline.Y)
            df[str_name].create_dataset('deg', data=[5])
            df[str_name].create_dataset('errors', data=spline.errors)
            df[str_name].create_dataset('tol', data=spline.tol)
            print(str_name, count)
            str_name = 'amp_l'+str(l)+'_m'+ str(-m)
            df.create_group(str_name)
            df[str_name].create_dataset('X', data=spline.X)
            df[str_name].create_dataset('Y', data=spline.Y)
            df[str_name].create_dataset('deg', data=[5])
            df[str_name].create_dataset('errors', data=spline.errors)
            df[str_name].create_dataset('tol', data=spline.tol)
            df.close()
            print(str_name, count)
            count = count + 1
            

df = h5py.File(file_name, 'a')
df.create_group('auxillary info')
df.close()
count = 3

for l in range(2, l_max+1):
    for m in range(0, l+1, 1):
        if m==0:
            temp_r = np.array(gw_core['rh_22'][keys[count]][:, 1])
            temp_i = np.array(gw_core['rh_22'][keys[count]][:, 2])
            time_ar = np.array(gw_core['rh_22'][keys[count]][:, 8])
            comp = temp_r + 1j*temp_i
            phase = np.angle(comp)
            df = h5py.File(file_name, 'a')
            str_name = 'phase_l'+str(l)+'_m'+ str(m)
            df.create_group(str_name)
            spline = romspline.ReducedOrderSpline(time_ar, phase, verbose=False)
            df[str_name].create_dataset('X', data=spline.X)
            df[str_name].create_dataset('Y', data=spline.Y)
            df[str_name].create_dataset('deg', data=[5])
            df[str_name].create_dataset('errors', data=spline.errors)
            df[str_name].create_dataset('tol', data=spline.tol)
            print(str_name, count)
            count = count + 1 
        else:
            temp_r = np.array(gw_core['rh_22'][keys[count]][:, 1])
            temp_i = np.array(gw_core['rh_22'][keys[count]][:, 2])
            time_ar = np.array(gw_core['rh_22'][keys[count]][:, 8])
            comp = temp_r + 1j*temp_i
            phase = np.angle(comp)
            df = h5py.File(file_name, 'a')
            str_name = 'phase_l'+str(l)+'_m'+ str(m)
            df.create_group(str_name)
            spline = romspline.ReducedOrderSpline(time_ar, phase, verbose=False)
            df[str_name].create_dataset('X', data=spline.X)
            df[str_name].create_dataset('Y', data=spline.Y)
            df[str_name].create_dataset('deg', data=[5])
            df[str_name].create_dataset('errors', data=spline.errors)
            df[str_name].create_dataset('tol', data=spline.tol)
            print(str_name, count)
            str_name = 'phase_l'+str(l)+'_m'+ str(-m)
            df.create_group(str_name)
            df[str_name].create_dataset('X', data=spline.X)
            df[str_name].create_dataset('Y', data=spline.Y)
            df[str_name].create_dataset('deg', data=[5])
            df[str_name].create_dataset('errors', data=spline.errors)
            df[str_name].create_dataset('tol', data=spline.tol)
            df.close()
            print(str_name, count)
            count = count + 1
            


new_file = h5py.File(file_name, 'r')
print(new_file.keys())
print(keys)

new_file.close()
