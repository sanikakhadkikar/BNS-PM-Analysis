#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  ligolw_pmrinj
#
#  Copyright 2017
#  James Clark <james.clark@ligo.org>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
"""
Generate a sim-inspiral table for waveforms from BNS Post-Merger Remnants
"""
#from __future__ import print_function
import os, sys
import h5py
import argparse
from subprocess import check_output
import pickle as pickle

import numpy as np
from scipy import random
import collections

import lal
from glue.ligolw import ligolw
from glue.ligolw import lsctables
from glue.ligolw import ilwd
from glue.ligolw.utils import process
from glue.ligolw import utils
from glue.lal import LIGOTimeGPS

import lalsimulation as lalsim

import pycbc
from pycbc import DYN_RANGE_FAC, pnutils
from pycbc.detector import Detector
from pycbc.inject import InjectionSet, legacy_approximant_name
from pycbc.types import TimeSeries, FrequencySeries, zeros
from pycbc.waveform import get_td_waveform, get_fd_waveform, taper_timeseries, spa_tmplt
from pycbc.filter import sigmasq, make_frequency_series
from pycbc.waveform import td_approximants, fd_approximants
from pycbc import psd as _psd

import matplotlib
matplotlib.use("Agg") # Needed to run on the CIT cluster


#def eprint(*args, **kwargs):
#    print(*args, file=sys.stderr, **kwargs)


# define a content handler
class LIGOLWContentHandler(ligolw.LIGOLWContentHandler):
    pass
lsctables.use_in(LIGOLWContentHandler)

def effective_distance(distance, inclination, f_plus, f_cross):
    return distance / np.sqrt( ( 1 + np.cos( inclination )**2 )**2 / 4 *
            f_plus**2 + np.cos( inclination )**2 * f_cross**2 )

# Object to store burst spectral parameters
burstParams = collections.namedtuple('burstParams', 
                   ['intHrss', 'extHrss', 'fchar', 'fpeak'])


def update_progress(progress):
    #print('\r\r[{0}] {1}'.format('#'*(progress/2)+' '*(50-progress/2), progress)),
    if progress == 100:
        print("\nDone")
    sys.stdout.flush()

def _empty_row(obj):
    """Create an empty sim_inspiral row where the columns have default values of
    0.0 for a float, 0 for an int, '' for a string. The ilwd columns have a
    default where the index is 0.
    """

    row = lsctables.SimInspiral()
    cols = lsctables.SimInspiralTable.validcolumns

    # populate columns with default values
    for entry in cols.keys():
        if cols[entry] in ['real_4','real_8']:
            setattr(row,entry,0.)
        elif cols[entry] == 'int_4s':
            setattr(row,entry,0)
        elif cols[entry] == 'lstring':
            setattr(row,entry,'')
        elif entry == 'process_id':
            row.process_id = ilwd.ilwdchar("sim_inspiral:process_id:0")
        elif entry == 'simulation_id':
            row.simulation_id = ilwd.ilwdchar("sim_inspiral:simulation_id:0")
        else:
            raise ValueError("Column %s not recognized." %(entry) )

        if entry in ['amp_order', 'phase_order', 'spin_order', 'tidal_order',
                'eccentricity_order']:
            setattr(row, entry, -1)

    return row

def uniform_dec(num):
    """
    Declination distribution: uniform in sin(dec). num controls the number of draws.
    """
    return (np.pi / 2.) - np.arccos(2 * random.random_sample(num) - 1)

def uniform_theta(num):
    """
    Uniform in cos distribution. num controls the number of draws.
    """
    return np.arccos(2 * random.random_sample(num) - 1)

def uniform_phi(num):
    """
    Uniform in (0, 2pi) distribution. num controls the number of draws.
    """
    return random.random_sample(num) * 2 * np.pi

def uniform_sky(num):
    """
    Get a set of (RA, declination, polarization) randomized appopriately to
    astrophysical sources isotropically distributed in the sky.
    """
    ra = uniform_phi(num)
    dec = uniform_dec(num)
    pol = uniform_phi(num)
    inc = uniform_dec(num)
    phase = uniform_phi(num)
    return ra, dec, pol, inc, phase


def volume_distributed_distances(num):
    """
    Get a set of event distances which is randomized uniformly in the volume.
    """
    return random.power(3, num) * self.distance 

def effective_distances(distance, geo_time, ra, dec, inclination, polarization):
    """
    Compute effective distances for everything in the HLVGT network

    Returns a dictionary with a tuple (eff_dist, end_time) for each IFO
    """
    instruments = {'H1':'h','L1':'l','V1':'v','G1':'g','T1':'t'}
    eff_dists = dict()

    for ifo in instruments.keys():

        # get Detector instance for IFO
        det = Detector(ifo)

        # get time delay to detector from center of the Earth
        time_delay = det.time_delay_from_earth_center(ra, dec, geo_time)
        local_time = geo_time + time_delay

        # get antenna pattern
        fp, fc = det.antenna_pattern(ra, dec, polarization, local_time)
        eff_distance = effective_distance(distance, inclination, fp, fc)

        eff_dists[ifo] = (eff_distance, local_time)
        
    return eff_dists

def hrss_to_energy(hrss, frequency, Deff, iota):
    """Convert hrss to energy in the narrow-band assumption"""
        
    Deff *= lal.PC_SI * 1e6
    Egw = 2*np.pi*2 * lal.C_SI**3 * \
         frequency**2 *hrss**2 * Deff**2
    Egw /= 5*lal.G_SI

    Egw /= lal.MSUN_SI*lal.C_SI*lal.C_SI
    
    return Egw

class population(object):
    """
    Container for BNS post-merger remnant signal population
    """

    def __init__(self, numrel_data, mtotal, tstart=None, tstop=None,
            signals_per_hour=None, min_distance=None, max_distance=None,
            posfile=None):

        self.numrel_data = numrel_data
        self.mtotal = mtotal
        self.tstart = tstart
        self.tstop = tstop
        self.rate = signals_per_hour
        if self.tstart==self.tstop:
            self.expnum = 1
        else:
            if self.rate is not None:
                self.expnum = int(np.ceil((self.tstop-self.tstart) * self.rate / 60.0 / 60.0))

        self.min_distance = min_distance
        self.max_distance = max_distance

    def geth5attr(self, attr):
        numrel_fileobj = h5py.File(self.numrel_data,'r')
        att = numrel_fileobj.attrs[attr]
        numrel_fileobj.close()
        return att

    def draw_random_times(self):
        """
        Draw set of uniformly distributed event times.
        """
        return random.randint(self.tstart, self.tstop, self.expnum) + random.rand(self.expnum)

    def draw_fixed_times(self,jitter=1.0):
        """
        Draw a set of regularly times at fixed intervals with a random jitter
        """

        if self.tstop == self.tstart:
            times = [self.tstart]
        else:
            interval = (self.tstop - self.tstart) / self.expnum
            times = np.array(map(lambda i: self.tstart + i*interval +
                jitter*(random.rand()-0.5), range(1,self.expnum+1)))

        return times
    
    def volume_distributed_distances(self):
        """
        Get a set of event distances which is randomized uniformly in volume.
        """
        distMin=self.min_distance
        distMax=self.max_distance

        d3min = distMin * distMin * distMin
        d3max = distMax * distMax * distMax
        deltad3 = d3max - d3min
        d3 = d3min + deltad3 * np.random.rand(self.expnum)

        return d3**(1./3)

    def uniform_sky(self):

        return uniform_sky(self.expnum)

    def posterior_sky_draws(self, posfile, ndraws=1000):
        """Use inspiral posterior draws for extrinsic parameters
        """
        #
        # Load and parse posterior samples file
        #
        print("Parsing posterior samples file")
        from pylal import bayespputils as bppu
        peparser = bppu.PEOutputParser('common')
        resultsObj = peparser.parse(open(posfile, 'r'))
        posterior = bppu.Posterior(resultsObj)

        #
        # Get MAP sample to waveform params
        #
        if ndraws>len(posterior):
            print >> sys.stderr, \
                    "not enough posterior samples for requested injections"

        ra  = np.concatenate(posterior['ra'].samples[:ndraws])
        dec = np.concatenate(posterior['dec'].samples[:ndraws])
        pol = np.concatenate(posterior['psi'].samples[:ndraws])
        #inc = np.concatenate(posterior['iota'].samples[:ndraws])
        inc = np.concatenate(posterior['theta_jn'].samples[:ndraws])
        distance = np.concatenate(posterior['distance'].samples[:ndraws])
        try:
            coa_phase = np.concatenate(posterior['phase'].samples[:ndraws])
        except:
            coa_phase = np.concatenate(posterior['phase_maxl'].samples[:ndraws])
        times = np.concatenate(posterior['h1_end_time'].samples[:ndraws])

        return ra, dec, pol, inc, distance, coa_phase, times


def pad_timeseries_to_integer_length(timeseries, sample_rate):
    """
    This function zero pads a time series so that its length is an integer
    multiple of the sampling rate.

    Padding is adding symmetically to the start and end of the time series.
    If the number of samples to pad is odd then the end zero padding will have
    one more sample than the start zero padding.
    """

    # calculate how many sample points needed to pad to get
    # integer second time series
    remainder = sample_rate - len(timeseries) % sample_rate
    start_pad = int(remainder / 2)
    end_pad = int(remainder - start_pad)

    # make arrays of zeroes
    start_array = np.zeros(start_pad)
    end_array = np.zeros(end_pad)

    # pad waveform with arrays of zeroes
    initial_array = np.concatenate([start_array,timeseries,end_array])
    return TimeSeries(initial_array, delta_t=timeseries.delta_t,
                      epoch=timeseries.start_time, dtype=timeseries.dtype)

def generate_strain(sim, ifo, delta_t, suppress_inspiral=False):
    """
    Generate strain TimeSeries for IFO 
    """

    # parse the sim_inspiral waveform column
    approx, _ = legacy_approximant_name(sim.waveform)
    print(sim.spin1x)
    hp, hc = get_td_waveform(sim, approximant=approx,
            f_lower=sim.f_lower, delta_t=delta_t)

    # zero pad polarizations to get integer second time series
    sample_rate = 1./delta_t
    hp = pad_timeseries_to_integer_length(hp, sample_rate)
    hc = pad_timeseries_to_integer_length(hc, sample_rate)


    # get Detector instance for IFO
    det = Detector(ifo)

    # get antenna pattern
    fp, fc = det.antenna_pattern(sim.longitude, sim.latitude,
            sim.polarization, sim.geocent_end_time)

    # calculate strain
    strain = fp*hp + fc*hc
    strain = pad_timeseries_to_integer_length(strain, sample_rate)

    # Window out the inspiral
    if suppress_inspiral:
        delay=0#int(1e-3 / delta_t)
        truncidx = np.argmax(abs(strain)) - delay
        strain.data[:truncidx] = 0.0
        truncidx = np.argmax(abs(hp)) - delay
        hp.data[:truncidx] = 0.0
        truncidx = np.argmax(abs(hc)) - delay
        hc.data[:truncidx] = 0.0

    # taper waveform
    hp = taper_timeseries(hp, tapermethod="TAPER_STARTEND")
    hc = taper_timeseries(hc, tapermethod="TAPER_STARTEND")
    strain = taper_timeseries(strain, tapermethod="TAPER_STARTEND")

    return strain, hp, hc

def compute_snr(sim, psd_dict, merger_phase, fmin=10.0, fmax=None,
        delta_t=1./8192, suppress_inspiral=False):
    """
    Compute network SNR for H,L using BayesWave PSDs in psd_path

    merger_phase determines whether to compute the SNRs for the pre-merger phase
    (where we just use the SPA horizon distance), or the SNRs for the
    post-merger phase where we actually generate the waveform.
    """

    snr_dict=dict()
    chars_dict=dict()
    ifos = psd_dict.keys()
    netsnr=0.0
    

    
    for i, ifo in enumerate(ifos):
        #print(psd_dict['H1'])
        #print(psd_dict.values())
        #Put psd_dict.values()[0] if using more than one detectors
        if fmax is None and list(psd_dict.values())[0] is not None:
           # fmax=min([0.5/delta_t,
           #     psd_dict['H1'].sample_frequencies().max()])
            fmax=4096.

        if merger_phase=='post':

            # Generate strains
            strain, hp, hc = generate_strain(sim, ifo=ifo, delta_t=delta_t,
                    suppress_inspiral=suppress_inspiral)


            # FFT strain
            strain_tilde = make_frequency_series(strain)
            hp_tilde = make_frequency_series(hp)
            hc_tilde = make_frequency_series(hc)

            # interpolate PSD to waveform delta_f
            if psd_dict[ifo] is not None:
                if psd_dict[ifo].delta_f != strain_tilde.delta_f:
                    psd_dict[ifo] = _psd.interpolate(
                                                psd_dict[ifo], strain_tilde.delta_f)

            # calculate sigma-squared SNR
            sigma_squared = float(sigmasq(
                                strain_tilde,
                                psd=psd_dict[ifo],
                                low_frequency_cutoff=fmin,
                                high_frequency_cutoff=fmax))

            # Burst parameters taken from Shourov Chatterji's thesis
            # psi(f) = amplitude spectrum |h(f)|, normalized to unity
            # fchar = first moment of frequency, treating psi(f) as a
            # probability distribution

            # signal characteristics as seen by <ifo>
            hrss_plus_sq = 0.5*float(sigmasq( hp, psd=None,
                low_frequency_cutoff=fmin,
                high_frequency_cutoff=fmax))
            hrss_cross_sq = 0.5*float(sigmasq( hc, psd=None,
                low_frequency_cutoff=fmin,
                high_frequency_cutoff=fmax))
            # intrinsic hrss: signal at geocenter
            int_hrss_sq = hrss_plus_sq + hrss_cross_sq

            # extrinsic: signal in detector x (including Fp,c)
            ext_hrss_sq = 0.5*float(sigmasq( strain_tilde, psd=None,
                low_frequency_cutoff=fmin,
                high_frequency_cutoff=fmax))


            psi_f = strain_tilde / np.sqrt(ext_hrss_sq)
            freqs=psi_f.sample_frequencies.data
            # XXX hardcoding
            peakidx = freqs>2000.  # only allow fpeak>2000 Hz

            norm = 2.0 * strain_tilde.delta_f
            psi_sq = abs(psi_f.data[freqs>=fmin])**2 * norm**2

            fchar = \
                    2*np.trapz(freqs[freqs>=fmin]*psi_sq,
                            freqs[freqs>=fmin])
            try:
                fpeak = freqs[peakidx][np.argmax(abs(strain_tilde)[peakidx])]
            except:
                fpeak = np.nan
            chars_dict['{}burstParams'.format(ifo)] = \
                    burstParams(np.sqrt(ext_hrss_sq), np.sqrt(int_hrss_sq),
                            fchar, fpeak)

            # Compute energy
            #Egw = hrss_to_energy(np.sqrt(ext_hrss_sq), fchar, getattr(sim,
            #    'eff_dist_'+ifo[0].lower()), sim.inclination)
            freqs = np.array(hp_tilde.sample_frequencies)
            idx = (freqs>=fmin)*(freqs<fmax)
            hiota = 2*hp_tilde.data*norm/(1+np.cos(sim.inclination)*np.cos(sim.inclination))

            dist = sim.distance*lal.PC_SI*1e6
            Egw = (4./5) * np.pi**2 * dist**2 
            Egw *= np.trapz((freqs[idx]*abs(hiota[idx]))**2,
                    freqs[idx])
            Egw *= (lal.C_SI**3) / lal.G_SI
            Egw /= (lal.MSUN_SI*lal.C_SI*lal.C_SI)

            if suppress_inspiral:
                print("Inspiral suppressed:"), Egw
            else: print("all HF:"), Egw*2


            # Compute SNR
            snr_dict[ifo]=np.sqrt(sigma_squared)


        elif merger_phase=='pre':

            chars_dict=np.nan
            if psd_dict[ifo] is None:
                snr_dict[ifo] = np.nan
            else:
                try:
                    # spa_templt is now broken because numpy does not accept
                    # floats as indices
                    horizon = spa_tmplt.spa_distance(psd_dict[ifo], sim.mass1, sim.mass2,
                            fmin, snr=8.)
                    snr_dict[ifo] = 8*horizon/getattr(sim, 'eff_dist_'+ifo[0].lower())
                except:
                    snr_dict[ifo] = np.nan

        else:
            raise ValueError("merger_phase must be one of 'pre' or 'post'")

        netsnr+=snr_dict[ifo]**2

    snr_dict['net']=np.sqrt(netsnr)

    return snr_dict, chars_dict

def rescale_signal(sim, psd_dict, merger_phase, Dmax=None, fmin=10., fmax=None,
        delta_t=1./8192, netsnr_threshold=None, netsnr_target=None,
        snr_threshold=None, suppress_inspiral=False):
    """
    Re-draw or rescale this source's distance such that the network SNR is above
    snr_threshold OR equal to snr_target
    """

    snr_dict, _ = compute_snr(sim, psd_dict, merger_phase, fmin=fmin,
                    fmax=fmax, delta_t=delta_t,
                    suppress_inspiral=suppress_inspiral)


    old_distance = np.copy(sim.distance)
    if netsnr_threshold is not None:
        # Re-draw distances until the SNR is greater than the threshold
        while snr_dict['net'] < netsnr_threshold:

            sim.distance=random.power(3)*Dmax
            # Re-scale effective distances
            for ifo in psd_dict.keys():
                attr='eff_dist_'+ifo[0].lower()
                setattr(sim, attr, getattr(sim,attr)*sim.distance/old_distance)
            old_distance = np.copy(sim.distance)

            snr_dict, _ = compute_snr(sim, psd_dict, merger_phase, fmin=fmin,
                    fmax=fmax, delta_t=delta_t)

    if snr_threshold is not None:
        # Re-draw distances until the SNR is greater than the threshold
        sngl_snrs = np.array([snr_dict[key] for key in snr_dict.keys() if
            key!='net'])
        while (sngl_snrs < snr_threshold).all():

            sim.distance=random.power(3)*Dmax
            # Re-scale effective distances
            for ifo in psd_dict.keys():
                attr='eff_dist_'+ifo[0].lower()
                setattr(sim, attr, getattr(sim,attr)*sim.distance/old_distance)
            old_distance = np.copy(sim.distance)

            snr_dict, _ = compute_snr(sim, psd_dict, merger_phase, fmin=fmin,
                    fmax=fmax, delta_t=delta_t)
            sngl_snrs = np.array([snr_dict[key] for key in snr_dict.keys() if
                key!='net'])

    if netsnr_target is not None:

        sim.distance=old_distance * snr_dict['net'] / netsnr_target

        # Re-scale effective distances
        for ifo in psd_dict.keys():
            attr='eff_dist_'+ifo[0].lower()
            setattr(sim, attr, getattr(sim,attr)*sim.distance/old_distance)

        snr_dict, _ = compute_snr(sim, psd_dict, merger_phase, fmin=fmin,
                fmax=fmax, delta_t=delta_t)

    return sim

def parse():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--verbose", default=False, action="store_true",
            help="""Instead of a progress bar, Print distances & SNRs to
            stdout""")

    parser.add_argument("--gps-start-time", metavar="GPSSTART", type=float,
            default=None, help="time to start injections", required=False)

    parser.add_argument("--gps-end-time", metavar="GPSEND", type=float,
            default=None, help="time to stop injections", required=False)

    parser.add_argument("--posterior-samples", metavar="POSSAMPS", type=str,
            default=None, help="cbc posterior_samples.dat", required=False)

    parser.add_argument("--jitter", metavar="JITTER", type=float, default=0.0,
            help="time window to jitter injections")

    parser.add_argument("--time-distribution", metavar="TDIST", type=str,
            default="fixed", help="time distribution (fixed or random)")

    parser.add_argument("--mtotal", metavar="MTOTAL", type=float, help="Total \
            mass for this waveform", required=True)

    parser.add_argument("--signals-per-hour", metavar="RATE", type=int,
            help="Number of signals to inject per hour")

    parser.add_argument("--seed", metavar="SEED", type=int, default=None,
            help="RNG seed")

    parser.add_argument("--min-distance", metavar="DISTANCE", type=float,
            default=1.0, help="Minimum distance in Mpc for volume \
            distribution")

    parser.add_argument("--max-distance", metavar="DISTANCE", type=float,
            default=200.0, help="Maximum distance in Mpc for volume \
            distribution")

    parser.add_argument("--numrel-data", metavar="NRHDF5", type=str, 
            help="Path to  file with simulation data", required=True)

    parser.add_argument("--f-lower-pre", metavar="PREFLOWER", type=float, 
            help="Frequency at which to start SNR for pre-merger waveforms",
            default=10.0, required=False)

    parser.add_argument("--f-lower-post", metavar="POSTFLOWER", type=float, 
            help="Frequency at which to start SNR for the post-merger waveform",
            default=1024.0, required=False)

    parser.add_argument("--srate-pre", metavar="PRESRATE", type=int, 
            help="Sample rate for pre-merger waveforms",
            default=2048, required=False)

    parser.add_argument("--srate-post", metavar="POSTSRATE", type=int, 
            help="Sample rate for post-merger waveforms",
            default=8192, required=False)

    parser.add_argument("--ifos", nargs=3, metavar="IFOS", type=str,
        required=True, 
        help="""IFOS to use for SNR calculations"""),

    parser.add_argument("--asd-files", nargs=3, metavar="ASDS", type=str,
        default=None, help="""Text files with ASDS.  Specified as
        --asd-files h1-file l1-file, etc.  Must be in the same order as
        --ifos""", required=False),

    parser.add_argument("--output", metavar="FILE.xml", type=str, 
            help="Name of output file", required=True)

    parser.add_argument("--fixed-pre-snr", metavar="PRESNR", type=float,
            help="""Scale all injection distances so that the pre-merger signal
            has this network SNR""", default=None)

    parser.add_argument("--fixed-highfreq-snr", metavar="HIGHFREQSNR", type=float,
            help="""Scale all injection distances so that the full HF signal
            has this network SNR""", default=None)

    parser.add_argument("--fixed-post-snr", metavar="POSTSNR", type=float,
            help="""Scale all injection distances so that the true post-merger
            signal (inspiral suppressed) has this network SNR""", default=None)

    parser.add_argument("--min-pre-snr", metavar="MINPRESNR", type=float,
            help="""Scale injection distances so that the pre-merger network
            SNR is larger than this value.""", default=None)

    parser.add_argument("--min-highfreq-snr", metavar="MINHIGHFREQSNR", type=float,
            help="""Scale injection distances so that the post-merger network
            SNR is larger than this value.""", default=None)

    parser.add_argument("--min-post-snr", metavar="MINPOSTSNR", type=float,
            help="""Scale all injection distances so that the true post-merger
            signal (inspiral suppressed) is larger than this network SNR""",
            default=None)


    parser.add_argument("--min-pre-single-snr", metavar="MINPRESINGLESNR",
            type=float, help="""Scale injection distances so that the pre-merger
            signal is louder than this in at least one instrument""",
            default=None)

    parser.add_argument("--min-highfreq-single-snr", metavar="MINHIGHFREQSINGLESNR",
            type=float, help="""Scale injection distances so that the
            high-frequency signal is louder than this in at least one
            instrument""", default=None)

    parser.add_argument("--min-post-single-snr", metavar="MINPOSTSINGLESNR",
            type=float, help="""Scale injection distances so that the post-merger
            signal is louder than this in at least one instrument""",
            default=None)

    parser.add_argument("--fixed-distance", metavar="DISTANCE", type=float,
            help="""Place all injections at this physical distance (still with
            random orientations/locations).""", default=None)

    parser.add_argument("--fixed-inclination", metavar="INCLINATION",
            type=float, help="""Place all injections at this inclination""",
            default=None)

    parser.add_argument("--fixed-ra", metavar="RA",
            type=float, help="""Place all injections at this RA (aka longitude
            in sim-inspiral)""",
            default=None)

    parser.add_argument("--fixed-dec", metavar="DEC",
            type=float, help="""Place all injections at this DEC (aka latitude
            in sim-inspiral)""",
            default=None)

    parser.add_argument("--fixed-pol", metavar="POLARISATION",
            type=float, help="""Place all injections at this polarisation""",
            default=None)

    parser.add_argument("--skip-snrs", default=False, action="store_true",
            help="""Skip SNR computations""")
    parser.add_argument("--dump-pickles", default=False, action="store_true",
            help="""Store waveform characteristics in pickles""")

    parser.add_argument("--trigger-time", default=None,
            help="""Time for a signal we want to avoid""")
    parser.add_argument("--trigger-win", default=5.,
            help="""Length of time around a signal we want to avoid""")

    parser.add_argument("--amp-order", default=0)



    opts = parser.parse_args()
    print(opts.ifos)
#   if not opts.skip_snrs:
#       if len(opts.ifos)>len(opts.asd_files):
#           raise ValueError("Requesting more IFOs than there are ASD files")

    return opts


pre_snr_dump = []
high_snr_dump = []
post_snr_dump = []
high_chars_dump = []
post_chars_dump = []

def main():

    instruments = {'H1':'h','L1':'l','V1':'v','G1':'g','T1':'t'}

    # Parse input
    opts = parse()

    # Sanity check sample rates:  if computing SNR, we can only generate
    # waveforms with a nyquist frequency less than or equal to the max PSD
    # frequency

    # Setup PSD dictionary
    psd_dict=dict()
    if not opts.skip_snrs:print("Loading noise curves for SNR calcns")
    for i,ifo in enumerate(opts.ifos):
        psd_dict[ifo] = None
        if not opts.skip_snrs and len(opts.asd_files):
            asd_data = np.loadtxt(opts.asd_files[i])
            delta_f=np.diff(asd_data[:,0])[0]
            if asd_data[0,0]!=0.0:
                print >> sys.stderr, "WARNING: ASD data does not start at 0 Hz"
                print >> sys.stderr, "  padding with large values"

                freqs = np.arange(0,asd_data[-1,0]+delta_f,delta_f)
                print('INFO: Freqs is set')
                psd = np.zeros(len(freqs))
                print('Starting PSD calc')
                psd[freqs>=asd_data[0,0]] = asd_data[:,1]*asd_data[:,1]
                psd[freqs<asd_data[0,0]] = 1e10
            else: 
                psd = asd_data[:,1]*asd_data[:,1]
            print('Variables set, starting storing them')
            psd_dict[ifo] = FrequencySeries(psd, delta_f=delta_f)

            if psd_dict[ifo].sample_frequencies.max()+delta_f < 0.5*opts.srate_post:
                print >> sys.stderr, "ERROR: IFO {0} max frequency: {0} Hz but {1} Hz waveform sample rate requested".format(
                    psd_dict[ifo].sample_frequencies.max(), opts.srate_post)
                sys.exit(-1)

    # Setup some global parameters
    if opts.seed is not None:
        random.seed(opts.seed)
    else:
        random.seed(int(check_output(["lalapps_tconvert", "now"])))

    mtotal = opts.mtotal
    print('Set till Mtotal')
    # -----------------------------------------------------------------
    # Initialise Source Population
    #
    if opts.posterior_samples is None:
        injections = population(opts.numrel_data, mtotal,
                tstart=opts.gps_start_time, tstop=opts.gps_end_time,
                signals_per_hour=opts.signals_per_hour,
                min_distance=opts.min_distance,
                max_distance=opts.max_distance)
    else:
        injections = population(opts.numrel_data, mtotal, 
                posfile=opts.posterior_samples)
    print('Codes still working')
    if opts.time_distribution == "random":
        geocentric_end_times = injections.draw_random_times()
    elif opts.time_distribution == "fixed":
        geocentric_end_times = injections.draw_fixed_times(jitter=opts.jitter)
    else:
        print >> sys.stdout, "time distribution not recognised"
        sys.exit(-1)
    print('Hang in there')
    if opts.posterior_samples is None:
        ra, dec, pol, inc, coa_phase = injections.uniform_sky()
        distance = injections.volume_distributed_distances()
    else:
        ra, dec, pol, inc, distance, coa_phase, times = \
                injections.posterior_sky_draws(opts.posterior_samples)

        if opts.time_distribution == "posterior":
            geocentric_end_times = np.copy(times)

    # update number of injections
    injections.expnum = len(geocentric_end_times)
    print('Almost there')

    # ALlow pinned params even if using posterior samples
    if opts.fixed_distance is not None:
        distance = opts.fixed_distance*np.ones(injections.expnum)

    if opts.fixed_inclination is not None:
        inc = opts.fixed_inclination*np.ones(injections.expnum)

    if opts.fixed_ra is not None:
        ra = opts.fixed_ra*np.ones(injections.expnum)

    if opts.fixed_dec is not None:
        dec = opts.fixed_dec*np.ones(injections.expnum)

    if opts.fixed_pol is not None:
        pol = opts.fixed_pol*np.ones(injections.expnum)
    

    # -----------------------------------------------------------------
    # Create and populate sim-inspiral table
    #
    print('starting table creation yay!')
    # Create a new XML document and sim_inspiral table
    xmldoc = ligolw.Document()
    lw = xmldoc.appendChild(ligolw.LIGO_LW())
    sim_table = lsctables.New(lsctables.SimInspiralTable)

    #
    # Intrinsic params from HDF5
    #
    mass1 = opts.mtotal*injections.geth5attr('mass1')
    mass2 = opts.mtotal*injections.geth5attr('mass2')
    spin1x = injections.geth5attr('spin1x')
    spin1y = injections.geth5attr('spin1y')
    spin1z = injections.geth5attr('spin1z')
    spin2x = injections.geth5attr('spin2x')
    spin2y = injections.geth5attr('spin2y')
    spin2z = injections.geth5attr('spin2z')

    # Loop over injection times
    print('Beginning loop over {} injection times'.format(injections.expnum))
    ids=range(injections.expnum)
    for t, geocentric_end_time in enumerate(geocentric_end_times):
        print(geocentric_end_time)
        if opts.trigger_time is not None:
            if abs(geocentric_end_time - opts.trigger_time)<opts.trigger_win: 
                print("injection too near trigger, skipping")
                continue

        if not opts.verbose:
            update_progress((t+1)*100/len(geocentric_end_times))

        # Create empty sim-inspiral row
        row = _empty_row(lsctables.SimInspiral)

        # Fill in IDs
        row.process_id = ilwd.ilwdchar("process:process_id:{0:d}".format(t))
        row.simulation_id = ilwd.ilwdchar("sim_inspiral:simulation_id:{0:d}".\
                format(ids[t]))
        #
        # Waveform Columns
        #
        #setattr(row, 'waveform', 'NR_hdf5pseudoFourPN')
        setattr(row, 'waveform', 'NR_hdf5pseudoFourPN')
        setattr(row, 'numrel_data', injections.numrel_data)
#       setattr(row, 'numrel_mode_max', injections.numrel_mode_max)
#       setattr(row, 'numrel_mode_min', injections.numrel_mode_min)
        setattr(row, 'amp_order', opts.amp_order)
#
        #
        # location / orientation
        #
        setattr(row, 'latitude', dec[t])
        setattr(row, 'longitude', ra[t])
        setattr(row, 'polarization', pol[t])
        setattr(row, 'inclination', inc[t])
        setattr(row, 'distance', distance[t])
        setattr(row, 'coa_phase', coa_phase[t])

        #
        # IFO-specific parameters (<ifo>_end_time and eff_dist_<ifo>)
        #
        row.set_time_geocent(LIGOTimeGPS(float(geocentric_end_time)))

        # populate IFO end time and eff_dist columns
        eff_dist_dict = effective_distances(distance[t], geocentric_end_time,
                ra[t], dec[t], inc[t], pol[t])

        for ifo in eff_dist_dict.keys():
            setattr(row, 'eff_dist_'+ifo[0].lower(), eff_dist_dict[ifo][0])
            setattr(row, ifo[0].lower()+'_end_time', int(eff_dist_dict[ifo][1]))
            setattr(row, ifo[0].lower()+'_end_time_ns',
                    int(eff_dist_dict[ifo][1] % 1 * 1e9))


        mchirp, eta = pnutils.mass1_mass2_to_mchirp_eta(mass1, mass2) 

        setattr(row, 'mass1', mass1)
        setattr(row, 'mass2', mass2)
        setattr(row, 'spin1x', spin1x)
        setattr(row, 'spin1y', spin1y)
        setattr(row, 'spin1z', spin1z)
        setattr(row, 'spin2x', spin2x)
        setattr(row, 'spin2y', spin2y)
        setattr(row, 'spin2z', spin2z)
        setattr(row, 'mchirp', mchirp)
        setattr(row, 'eta', eta)

        #
        # Waveform generation details
        #
        setattr(row, 'f_lower', opts.f_lower_post)
        setattr(row, 'taper', 'TAPER_START')
        setattr(row, 'source', 'BNS')


        if not opts.skip_snrs:
            #
            # Characterize post-merger SNR (and rescale if desired)
            #

            # Pre-merger SNR
            pre_snr_dict, _ = compute_snr(row, psd_dict, merger_phase='pre',
                    fmin=opts.f_lower_pre)

            # high-freq SNR
            highfreq_snr_dict, _ = compute_snr(row, psd_dict, merger_phase='post',
                    delta_t=1./opts.srate_post, fmin=opts.f_lower_post)

                # Post-merger SNR
            post_snr_dict, _ = compute_snr(row, psd_dict, merger_phase='post',
                    delta_t=1./opts.srate_post, fmin=opts.f_lower_post,
                    suppress_inspiral=True)

            if opts.verbose:
                print("\n--------{0} of {1}---------".format(t+1,
                        len(geocentric_end_times)))
                print("physical distance=", row.distance)
                print("eff_dist_h={0}, eff_dist_l={1}".format(row.eff_dist_h,
                        row.eff_dist_l))
                print("initial pre-merger netSNR: {}".format(pre_snr_dict['net']))
        #        print "(SNR-dict: )", pre_snr_dict
                print("initial HF netSNR: {}".format(highfreq_snr_dict['net']))
        #        print "(SNR-dict: )\n", highfreq_snr_dict
                print("initial post-merger netSNR: {}".format(post_snr_dict['net']))
        #        print "(SNR-dict: )\n", post_snr_dict
            

            #
            # Rescale SNRs to targets
            #

            # --- If pre-merger network SNR < threshold, rescale

            if (opts.min_pre_snr is not None) and \
                    (pre_snr_dict['net']<opts.min_pre_snr): 

                if opts.verbose: 
                    print("...pre-merger netSNR<threshold, rescaling distance...")

                row = rescale_signal(row, psd_dict, merger_phase='pre',
                        Dmax=opts.max_distance, fmin=opts.f_lower_pre,
                        netsnr_threshold=opts.min_pre_snr)


            # --- If all pre-merger single SNRs < threshold, 
            #      rescale until (at least) one is > threshold

            sngl_snrs = np.array([pre_snr_dict[key] for key in pre_snr_dict.keys()
                if key!='net'])
            if (opts.min_pre_single_snr is not None) and \
                    ( (sngl_snrs<opts.min_pre_single_snr).all() ): 

                if opts.verbose: 
                    print("...pre-merger single SNR<threshold, rescaling distance...")

                row = rescale_signal(row, psd_dict, merger_phase='pre',
                        Dmax=opts.max_distance, fmin=opts.f_lower_pre,
                        snr_threshold=opts.min_pre_single_snr)

            # --- If HF network SNR < threshold, rescale
            if (opts.min_highfreq_snr is not None) and \
                    (highfreq_snr_dict['net']<opts.min_highfreq_snr): 


                if opts.verbose: 
                    print("...highfreq-merger netSNR<threshold, rescaling distance...")

                row = rescale_signal(row, psd_dict, merger_phase='post',
                        Dmax=opts.max_distance, fmin=opts.f_lower_post,
                        netsnr_threshold=opts.min_highfreq_snr)
                

            # --- If all HF single SNRs < threshold, 
            #      rescale until (at least) one is > threshold
            sngl_snrs = np.array([highfreq_snr_dict[key] for key in highfreq_snr_dict.keys()
                if key!='net'])
            if (opts.min_highfreq_single_snr is not None) and \
                    ( (sngl_snrs<opts.min_highfreq_single_snr).all() ): 

                if opts.verbose: 
                    print("...high-freq single SNR<threshold, rescaling distance...")

                row = rescale_signal(row, psd_dict, merger_phase='post',
                        Dmax=opts.max_distance, fmin=opts.f_lower_post,
                        snr_threshold=opts.min_highfreq_single_snr)

            # --- If postmerger network SNR < threshold, rescale
            if (opts.min_post_snr is not None) and \
                    (post_snr_dict['net']<opts.min_post_snr): 

                if opts.verbose: 
                    print("...post-merger netSNR<threshold, rescaling distance...")

                row = rescale_signal(row, psd_dict, merger_phase='post',
                        Dmax=opts.max_distance, fmin=opts.f_lower_post,
                        netsnr_threshold=opts.min_post_snr,
                        suppress_inspiral=True)

            # --- If all postmerger single SNRs < threshold, 
            #      rescale until (at least) one is > threshold

            sngl_snrs = np.array([post_snr_dict[key] for key in post_snr_dict.keys()
                if key!='net'])
            if (opts.min_post_single_snr is not None) and \
                    ( (sngl_snrs<opts.min_post_single_snr).all() ): 

                if opts.verbose: 
                    print("...postmerger single SNR<threshold, rescaling distance...")

                row = rescale_signal(row, psd_dict, merger_phase='post',
                        Dmax=opts.max_distance, fmin=opts.f_lower_post,
                        snr_threshold=opts.min_post_single_snr,
                        suppress_inspiral=True)

     
            # --- Fix the pre-merger SNR
            if opts.fixed_pre_snr is not None: 

                row = rescale_signal(row, psd_dict, merger_phase='pre',
                        fmin=opts.f_lower_pre, delta_t=1./opts.srate_post,
                        netsnr_target=opts.fixed_pre_snr)

            # --- Fix the HF SNR
            if opts.fixed_highfreq_snr is not None: 

                row = rescale_signal(row, psd_dict, merger_phase='post',
                        fmin=opts.f_lower_post, delta_t=1./opts.srate_post,
                        netsnr_target=opts.fixed_highfreq_snr)

            # --- Fix the Post-merger SNR
            if opts.fixed_post_snr is not None: 

                row = rescale_signal(row, psd_dict, merger_phase='post',
                        fmin=opts.f_lower_post, delta_t=1./opts.srate_post,
                        netsnr_target=opts.fixed_post_snr, suppress_inspiral=True)

            # --- Finally, re-compute all SNRs at rescaled distances
            pre_snr_dict, _ = compute_snr(row, psd_dict, merger_phase='pre',
                    fmin=opts.f_lower_pre)
            pre_snr_dump.append(pre_snr_dict)

            highfreq_snr_dict, highfreq_chars_dict = compute_snr(row, psd_dict, merger_phase='post',
                    delta_t=1./opts.srate_post, fmin=opts.f_lower_post)
            high_snr_dump.append(highfreq_snr_dict)
            high_chars_dump.append(highfreq_chars_dict)

            post_snr_dict, post_chars_dict = compute_snr(row, psd_dict, merger_phase='post',
                    delta_t=1./opts.srate_post, fmin=opts.f_lower_post,
                    suppress_inspiral=True)
            post_snr_dump.append(post_snr_dict)
            post_chars_dump.append(post_chars_dict)

            if opts.verbose:
                print("\nphysical distance=", row.distance)
                print("eff_dist_h={0}, eff_dist_l={1}".format(row.eff_dist_h,
                        row.eff_dist_l))
                print("final pre-merger netSNR: {}".format(pre_snr_dict['net']))
                print("final high-freq netSNR: {}".format(highfreq_snr_dict['net']))
                print("final post-merger netSNR: {}".format(post_snr_dict['net']))
                print( "--- intrinsic high-freq params")
                print("H1 hrss: {}".format(highfreq_chars_dict['H1burstParams'].intHrss))
                print("H1 fpeak: {}".format(highfreq_chars_dict['H1burstParams'].fpeak))
                print("H1 fchar: {}".format(highfreq_chars_dict['H1burstParams'].fchar))
                print("L1 hrss: {}".format(highfreq_chars_dict['L1burstParams'].intHrss))
                print("L1 fpeak: {}".format(highfreq_chars_dict['L1burstParams'].fpeak))
                print("L1 fchar: {}".format(highfreq_chars_dict['L1burstParams'].fchar))
                print("--- intrinsic remnant params:")
                print("H1 hrss: {}".format(post_chars_dict['H1burstParams'].intHrss))
                print("H1 fpeak: {}".format(post_chars_dict['H1burstParams'].fpeak))
                print("H1 fchar: {}".format(post_chars_dict['H1burstParams'].fchar))
                print("L1 hrss: {}".format(post_chars_dict['L1burstParams'].extHrss))
                print("L1 fpeak: {}".format(post_chars_dict['L1burstParams'].fpeak))
                print("L1 fchar: {}".format(post_chars_dict['L1burstParams'].fchar))

            # Store SNRs in alpha and beta for pre and post, respectively
            setattr(row, 'alpha', pre_snr_dict['net'])
            setattr(row, 'beta', highfreq_snr_dict['net'])
            setattr(row, 'alpha1', post_snr_dict['net'])

        # End if for SNR computation


        # Append this row to the sim_inspiral table
        sim_table.append(row)
 

    # -----------------------------------------------------------------
    # Create and write the XML document
    #
    if not opts.skip_snrs and opts.dump_pickles:
        pickle.dump(pre_snr_dump, open(opts.output.replace('.xml.gz','_preSNR.pickle'),'wb'))
        pickle.dump(high_snr_dump, open(opts.output.replace('.xml.gz','_highSNR.pickle'),'wb'))
        pickle.dump(post_snr_dump, open(opts.output.replace('.xml.gz','_postSNR.pickle'),'wb'))
        pickle.dump(high_chars_dump, open(opts.output.replace('.xml.gz','_highChars.pickle'),'wb'))
        pickle.dump(post_chars_dump, open(opts.output.replace('.xml.gz','_postChars.pickle'),'wb'))

    lw.appendChild(sim_table)
    # Write file
    utils.write_filename(xmldoc, opts.output,
            gz=opts.output.endswith("gz"), verbose=True)


    return xmldoc, sim_table, injections

if __name__ == "__main__":
    xmldoc, sim_table, injections = main()
