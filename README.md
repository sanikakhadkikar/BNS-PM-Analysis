# BNS-PM-Analysis
This is a code written to analyse Computational Relativity(CoRe) database's waveforms using Bayesian inference frameworks. The repository is organised as follows: 
- *file_converter.py*: A simple python code to convert CoRe waveforms to the LIGO compatible NRHDF5 format. 
- *xml_converter.py*: Used to create the .xml file required to accompany the frame files during injections 
- *gw_waveform_analyser.py*: A post processing script for sanity checks 

## Module requirements: 
The module listed below are what will be required to use this code sequence:
-Romspline 
-PyCBC
-LaLSuite 
-H5py
These can be imported by running: 
```
pip install romspline pycbc lalsuite h5py
```
