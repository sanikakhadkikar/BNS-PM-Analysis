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
df.attrs['NR-group'] = 'CoRe'
df.attrs['type'] = 'non spinning'
df.attrs['name'] = 'BHBlp_1.250_1.250_0.00_0.00_0.050'
df.attrs['object1'] = 'NS'
df.attrs['object2'] = 'NS'
df.attrs['mass1'] = 1.25
df.attrs['mass2'] = 1.25
df.attrs['eta'] = 0.25
df.attrs['spin1x'] = 0
df.attrs['spin1y'] = 0
df.attrs['spin1z'] = 0
df.attrs['spin2x'] = 0
df.attrs['spin2y'] = 0
df.attrs['spin2z'] = 0
df.attrs['LNhatx'] = 0
df.attrs['LNhaty'] = 0
df.attrs['LNhatz'] = 1
df.attrs['nhatx'] = 1
df.attrs['nhaty'] = 0
df.attrs['nhatz'] = 0
df.attrs['f_lower_at_1MSUN'] = 0.0498819129216


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
