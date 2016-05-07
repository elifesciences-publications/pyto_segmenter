'''Classes and methods for segmentation of spherical objects within cells.'''

#### WARNING: THIS SCRIPT IS CURRENTLY INCOMPLETE!!! ####

## IMPORT DEPENDENCIES
import os
import sys
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.morphology import watershed
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import distance_transform_edt

class PexSegmentObj:
    '''An object containing information for segmented peroxisomes.'''
    def __init__(self, f_directory, filename, raw_img, gaussian_img, mode,
                 mode_params = '', threshold_img, dist_map,
                 smooth_dist_map, maxima, labs, watershed_output, 
                 segmentation_log, volumes):
        '''Initialize the PexSegmentObj with segmentation data.'''
        self.log = segmentation_log
        self.log.append('creating PexSegmentObj...')
        self.f_directory = f_directory
        self.filename = filename
        self.raw_img = raw_img.astype('uint16')
        self.gaussian_img = gaussian_img.astype('uint16')
        self.mode = mode
        self.mode_params = mode_params
        self.threshold_img = threshold_img.astype('uint16')
        self.dist_map = dist_map.astype('uint16')
        self.smooth_dist_map = smooth_dist_map.astype('uint16')
        self.maxima = maxima.astype('uint16')
        self.labs = labs.astype('uint16')
        self.watershed_output = watershed_output.astype('uint16')
        self.slices = self.raw_img.shape[0]
        self.height = self.raw_img.shape[1]
        self.width = self.raw_img.shape[2]
        self.log = segmentation_log
        self.obj_nums = np.unique(watershed_output)
        self.npexs = len(self.obj_nums)-1
        self.volumes = volumes # in units of pixels.

    def __repr__(self):
        return 'PytoSegmentObj '+ self.filename

## TODO: FINISH IMPLEMENTING THIS CLASS.

class PexSegmenter:
    
    def __init__(self,filename, mode = 'threshold', **kwargs):
        self.log = []
        self.filename = filename
        self.mode = mode
        if mode == 'threshold':
            self.threshold = kwargs.get('threshold',float('nan'))
            if np.isnan(self.threshold):
                raise ValueError('A threshold argument must be provided to segment with a constant threshold.')
        if mode == 'bg_scaled':
            self.cells = kwargs.get('cells', '')
            self.bg_diff = kwargs.get('bg_diff',float('nan'))
            if self.cells == '':
                raise ValueError('A PytoSegmentObj containing segmented cells is required if mode == bg_scaled.')
            if np.isnan(self.bg_diff):
                raise ValueError('a bg_diff argument is needed if mode == bg_scaled.')
    
    def segment(self):
        '''Segment peroxisomes within the image.'''
        starttime = time.time() # begin timing
        f_directory = os.getcwd()
        # data import
        self.log.append('reading' + self.filename)
        raw_img = io.imread(self.filename)
        self.log.append('raw image imported.')
        # gaussian filter assuming 100x objective and 0.2 um slices
        self.log.append('performing gaussian filtering...')
        gaussian_img = gaussian_filter(raw_img, [1,1,1])
        self.log.append('cytosolic image smoothed.')
        self.log.append('preprocessing complete.')
        # binary thresholding and cleanup
        self.log.append('thresholding...')
        threshold_img = np.copy(gaussian_img)
        if self.mode == 'threshold':
           self.log.append('mode = threshold.')
           threshold_img[threshold_img < self.threshold] = 0
           threshold_img[threshold_img > 0] = 1
           self.log.append('thresholding complete.')
        if self.mode == 'bg_scaled':
            pass # TODO: IMPLEMENT THIS METHOD.
        # distance and maxima transformation to find objects
        # next two steps assume 100x objective and 0.2 um slices
        self.log.append('generating distance map...')
        dist_map = distance_transform_edt(threshold_img, sampling = (2,1,1))
        self.log.append('distance map complete.')
        self.log.append('smoothing distance map...')
        smooth_dist = gaussian_filter(dist_map, [1,2,2])
        self.log.append('distance map smoothed.')
        self.log.append('identifying maxima...')
        max_strel = generate_binary_structure(3,2)
        maxima = maximum_filter(smooth_dist,
                                max_strel) == smooth_dist
        # clean up background and edges
        bgrd_3d = smooth_dist == 0
        eroded_bgrd = binary_erosion(bgrd_3d, structure = max_strel,
                                     border_value = 1)
        maxima = np.logical_xor(maxima, eroded_bgrd)
        self.log.append('maxima identified.')
        # watershed segmentation
        labs = self.watershed_labels(maxima)
        self.log.append('watershedding...')
        peroxisomes = watershed(-smooth_dist, labs, mask = threshold_img)
        self.log.append('watershedding complete.')
        self.log.append('filtering out too-large and too-small objects...')
        obj_nums, volumes = np.unique(peroxisomes, return_counts = True)
        
        # the following code, in one line, eliminates all objects that are
        # either fewer than 5 or greater than 2500 pixels in volume.
        peroxisomes[np.in1d(peroxisomes, 
                            obj_nums[np.bitwise_or(volumes < 5,volumes >
                                                    2500)].astype('int')).reshape(peroxisomes.shape)]== 0
        obj_nums, volumes = np.unique(peroxisomes, return_counts = True)
        if self.mode == 'threshold':
            return PexSegmentObj(f_directory, self.filename, raw_img,
                                 gaussian_img, self.mode, self.threshold,
                                 threshold_img, dist_map, smooth_dist, maxima,
                                 labs, peroxisomes)
        elif self.mode == 'bg_scaled':
            return PexSegmentObj(f_directory, self.filename, raw_img,
                                 gaussian_img, self.mode, self.bg_diff,
                                 threshold_img, dist_map, smooth_dist, maxima,
                                 labs, peroxisomes, self.log, volumes)

    ## HELPER METHODS ##
    def watershed_labels(self, maxima_img):
        '''Takes a boolean array with maxima labeled as true pixels
        and returns an array with maxima numbered sequentially.'''
        
        max_z, max_y, max_x = np.nonzero(maxima_img)
        
        label_output = np.zeros(maxima_img.shape)
        
        for i in range(0,len(max_y)):
            label_output[max_z[i],max_y[i],max_x[i]] = i+1
        
        return(label_output)
