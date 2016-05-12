'''Classes and methods for segmentation of spherical objects within cells.'''

#### WARNING: THIS SCRIPT IS CURRENTLY INCOMPLETE!!! ####

## IMPORT DEPENDENCIES
import matplotlib
matplotlib.use('Agg')
import os
import sys
import pickle
import time
import numpy as np
from skimage import io
from skimage.morphology import watershed
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.morphology import distance_transform_edt, binary_dilation
import matplotlib.pyplot as plt


class PexSegmentObj:
    '''An object containing information for segmented peroxisomes.'''
    def __init__(self, f_directory, filename, raw_img, gaussian_img, mode,
                 threshold_img, dist_map,
                 smooth_dist_map, maxima, labs, watershed_output, 
                 segmentation_log, volumes, mode_params = ''):
        '''Initialize the PexSegmentObj with segmentation data.'''
        self.log = segmentation_log
        self.log.append('creating PexSegmentObj...')
        self.f_directory = f_directory
        self.filename = os.path.basename(filename)
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
        return 'PexSegmentObj '+ self.filename

    def plot_raw_img(self,display = False):
        self.plot_stack(self.raw_img, colormap = 'gray')
        if display == True:
            plt.show()
    def plot_gaussian_img(self, display = False):
        self.plot_stack(self.gaussian_img, colormap = 'gray')
        if display == True:
            plt.show()
    def plot_threshold_img(self, display = False):
        self.plot_stack(self.threshold_img, colormap = 'gray')
        if display == True:
            plt.show()
    def plot_dist_map(self, display = False):
        self.plot_stack(self.dist_map)
        if display == True:
            plt.show()
    def plot_smooth_dist_map(self, display = False):
        self.plot_stack(self.smooth_dist_map)
        if display == True:
            plt.show()
    def plot_maxima(self, display = False):
        vis_maxima = binary_dilation(self.maxima,
                                     structure = np.ones(shape = (1,5,5)))
        masked_maxima = np.ma.masked_where(vis_maxima == 0, vis_maxima)
        self.plot_maxima_stack(masked_maxima, self.smooth_dist_map)
        if display == True:
            plt.show()
    def plot_watershed(self, display = False):
        self.plot_stack(self.watershed_output)
        if display == True:
            plt.show()

    def output_images(self):
        '''Write all images to a new subdirectory.
        
        Write all images associated with the PexSegmentObj to a new
        directory. Name that directory according to the filename of the initial
        image that the object was derived from. This new directory should be a
        subdirectory to the directory containing the original raw image.
        '''
        os.chdir(self.f_directory)
        if not os.path.isdir(self.f_directory + '/' +
                             self.filename[0:self.filename.index('.')]):
            self.log.append('creating output directory...')
            os.mkdir(self.f_directory + '/' +
                     self.filename[0:self.filename.index('.')])
        os.chdir(self.f_directory + '/' +
                 self.filename[0:self.filename.index('.')])
        self.log.append('writing images...')
        io.imsave('raw_'+self.filename, self.raw_img)
        io.imsave('gaussian_'+self.filename, self.gaussian_img)
        io.imsave('threshold_'+self.filename, self.threshold_img)
        io.imsave('dist_'+self.filename, self.dist_map)
        io.imsave('smooth_dist_'+self.filename,self.smooth_dist_map)
        io.imsave('maxima_'+self.filename,self.maxima)
        io.imsave('wshed_'+self.filename,self.watershed_output)
    def output_plots(self):
        '''Write PDFs of slice-by-slice plots.
        
        Output: PDF plots of each image within PexSegmentObj in a directory
        named for the original filename they were generated from. Plots are
        generated using the plot_stack method and plotting methods defined
        here.
        '''
        os.chdir(self.f_directory)
        if not os.path.isdir(self.f_directory + '/' +
                             self.filename[0:self.filename.index('.')]):
            self.log.append('creating output directory...')
            os.mkdir(self.f_directory + '/' +
                     self.filename[0:self.filename.index('.')])
        os.chdir(self.f_directory + '/' +
                 self.filename[0:self.filename.index('.')])
        self.log.append('saving plots...')
        self.plot_raw_img()
        plt.savefig('praw_'+self.filename[0:self.filename.index('.')]+'.pdf')
        self.plot_gaussian_img()
        plt.savefig('pgaussian_' +
                    self.filename[0:self.filename.index('.')]+'.pdf')
        self.plot_threshold_img()
        plt.savefig('pthreshold_' +
                    self.filename[0:self.filename.index('.')]+'.pdf')
        plt.savefig('pdist_' +
                    self.filename[0:self.filename.index('.')]+'.pdf')
        self.plot_smooth_dist_map()
        plt.savefig('psmooth_dist_' + 
                    self.filename[0:self.filename.index('.')]+'.pdf')
        self.plot_maxima()
        plt.savefig('pmaxima_' +
                    self.filename[0:self.filename.index('.')]+'.pdf')
        self.plot_watershed()
        plt.savefig('pwshed_' +
                    self.filename[0:self.filename.index('.')]+'.pdf')
    def pickle(self):
        '''pickle the PexSegmentObj for later loading.'''
        if not os.path.isdir(self.f_directory + '/' +
                             self.filename[0:self.filename.index('.')]):
            self.log.append('creating output directory...')
            os.mkdir(self.f_directory + '/' + 
                     self.filename[0:self.filename.index('.')])
        os.chdir(self.f_directory + '/' + 
                 self.filename[0:self.filename.index('.')])
        with open('pickled_' +
                  self.filename[0:self.filename.index('.')] + 
                  '.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    def output_all(self):
        os.chdir(self.f_directory)
        if not os.path.isdir(self.f_directory + '/' +
                             self.filename[0:self.filename.index('.')]):
            os.mkdir(self.f_directory + '/' +
                     self.filename[0:self.filename.index('.')])
        os.chdir(self.f_directory + '/' +
                 self.filename[0:self.filename.index('.')])
        self.log.append('outputting all data...')
        self.output_plots()
        self.output_images()
        self.mk_log_file('log_'+self.filename[0:self.filename.index('.')]+'.txt')
        self.pickle()
    def output_slim(self):
        '''output the slimmed object.'''
        # TODO: IMPLEMENT THIS METHOD (WILL REQUIRE CREATING OUTPUT_SLIM_PLOTS
        # AND OUTPUT_IMAGES_SLIM AS WELL; OR, I COULD CREATE A SLIM ATTRIBUTE
        # AND ADJUST THESE METHODS TO CHANGE DEPENDING UPON THE STATE OF THE
        # SLIM ATTRIBUTE)
        pass

    ## HELPER METHODS ##

    def plot_stack(self, stack_arr, colormap='jet'):
        ''' Create a matplotlib plot with each subplot containing a slice.
        
        Keyword arguments:
        stack_arr: a numpy ndarray containing pixel intensity values.
        colormap: the colormap to be used when displaying pixel
                  intensities. defaults to jet.
        
        Output: a pyplot object in which each slice from the image array
                is represented in a subplot. subplots are 4 columns
                across (when 4 or more slices are present) with rows to
                accommodate all slices.
        '''

        nimgs = stack_arr.shape[0] # z axis of array dictates number of slices
        # plot with 4 imgs across

        # determine how many rows and columns of images there are

        if nimgs < 5:
            f, axarr = plt.subplots(1,nimgs)

            for i in range(0,nimgs):
                axarr[i].imshow(stack_arr[i,:,:], cmap=colormap)
                axarr[i].xaxis.set_visible(False)
                axarr[i].yaxis.set_visible(False)
            f.set_figwidth(16)
            f.set_figheight(4)        

        else:
            f, axarr = plt.subplots(int(np.ceil(nimgs/4)),4)

            for i in range(0,nimgs):
                r = int(np.floor(i/4))
                c = int(i % 4)
                axarr[r,c].imshow(stack_arr[i,:,:], cmap=colormap)
                axarr[r,c].xaxis.set_visible(False)
                axarr[r,c].yaxis.set_visible(False)

            if nimgs%4 > 0:
                r = int(np.floor(nimgs/4))

                for c in range(nimgs%4,4):
                    axarr[r,c].axis('off')

            f.set_figwidth(16)
            f.set_figheight(4*np.ceil(nimgs/4))

    def mk_log_file(self, fname):
        '''Write the log file list to a text file.
        kwargs:
            fname: filename to write to.
            '''
        self.log.append('making log file...')
        with open(fname, 'w') as f:
            for s in self.log:
                f.write(s + '\n')
            f.write('number of slices: ' + str(self.slices) + '\n')
            f.write('width: ' + str(self.width) + '\n')
            f.write('height: ' + str(self.height) + '\n')
            f.write('threshold parameters: ' + str(self.mode_params) + '\n')
            f.write('number of objects: ' + str(self.npexs) + '\n')
            f.write('volumes of objects in pixels: ' + 
                    str(self.volumes) + '\n')
            f.close()
    
    def slim(self):
        '''remove all of the processing intermediates from the object, leaving
        only the core information required for later analysis. primarily
        intended for use when doing batch analysis of multiple images, and
        combining PexSegmentObj instances with instances of other types of
        objects segmented in a different fluorescence channel.
        '''
        del self.raw_img
        del self.gaussian_img
        del self.threshold_img
        del self.dist_map
        del self.smooth_dist_map
        del self.maxima
        del self.labs

        return self

    def plot_maxima_stack(self, masked_max, smooth_dist):

        ''' Creates a matplotlib plot object in which each slice from the image
        is displayed as a single subplot, in a 4-by-n matrix (n depends upon
        the number of slices in the image)'''

        nimgs = masked_max.shape[0] # z axis of array dictates number of slices
        # plot with 4 imgs across

        # determine how many rows and columns of images there are

        if nimgs < 5:
            f, axarr = plt.subplots(1,nimgs)

            for i in range(0,nimgs):
                axarr[i].imshow(smooth_dist[i,:,:], cmap='gray')
                axarr[i].imshow(masked_max[i,:,:], cmap='autumn')
                axarr[i].xaxis.set_visible(False)
                axarr[i].yaxis.set_visible(False)
            f.set_figwidth(16)
            f.set_figheight(4)

        else:
            f, axarr = plt.subplots(int(np.ceil(nimgs/4)),4)

            for i in range(0,nimgs):
                r = int(np.floor(i/4))
                c = int(i%4)
                axarr[r,c].imshow(smooth_dist[i,:,:], cmap='gray')
                axarr[r,c].imshow(masked_max[i,:,:], cmap='autumn')
                axarr[r,c].xaxis.set_visible(False)
                axarr[r,c].yaxis.set_visible(False)

            if nimgs%4 > 0:
                r = int(np.floor(nimgs/4))

                for c in range(nimgs%4, 4):
                    axarr[r,c].axis('off')

            f.set_figwidth(16)
            f.set_figheight(4*np.ceil(nimgs/4))

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
                raise ValueError('A CellSegmentObj containing segmented cells is required if mode == bg_scaled.')
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
            self.log.append('mode = background-scaled.')
            self.thresholds = {}
            threshold_img = np.zeros(shape = raw_img.shape)
            for i in unique(self.cells):
                if i == 0:
                    pass
                else:
                    self.log.append('thresholding cell ' + str(i))
                    cell_median = np.median(raw_img[cells == i])
                    threshold_img[cells == i & 
                                  raw_img > cell_median + self.bg_diff] = 1
                # TODO: remember that after watershedding i'll have to find
                # peroxisomes that contact the edge of the segmented cell, and
                # if they do, grow them to fill the 6-connected 3d region of
                # pixels above the threshold value. this requires me to store
                # the cell that each peroxisome belongs to, as well as the
                # threshold value applied for each cell. i can implement the
                # cell assignment later when i get to watershedding, but i need
                # to store the cutoffs now.
                    self.thresholds[i] = cell_median + self.bg_diff #store val
            self.log.append('thresholding complete.')
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
                                footprint = max_strel) == smooth_dist
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
        if self.mode == 'bg_scaled':
            edge_struct = generate_binary_structure(3,1)
            self.c_edges = {}
            self.cellnums = [x for x in np.unique(self.cells) if x != 0]
            self.log.append('finding edges of cells...')
            for i in self.cellnums:
                self.c_edges[i] = np.logical_xor(self.cells == i,
                                                      binary_erosion(self.cells
                                                                     == i,
                                                                     edge_struct))
            self.log.append('cell edges found.')
            self.primary_objs = [x for x in np.unique(peroxisomes) if x != 0]
            self.assigned_cells = {}
            self.on_edge = {}
            for obj in primary_objs:
                self.assigned_cells[obj] = (self.cells[labs == obj])
                obj_mask = peroxisomes == obj
                obj_edge = np.logical_xor(obj_mask, 
                                          binary_erosion(obj_mask,
                                                         edge_struct))
                # test if the object's edge and its cell's edge overlap
                if np.any(np.logical_and(obj_edge,
                                         self.c_edges[self.assigned_cells[obj]])):
                    self.on_edge[obj] = True
                    new_obj = obj_mask
                    search_obj = obj_mask
                    tester = 0
                    while tester == 0:
                        grown_obj = binary_dilation(search_obj, edge_struct)
                        new_px = np.logical_xor(grown_obj, new_obj)
                        if np.any(gaussian_img[new_px] > self.thresholds[obj]):
                            to_add = np.logical_and(new_px, gaussian_img >
                                                    self.thresholds[obj])
                            new_obj = np.logical_or(new_obj, to_add)
                            search_obj = to_add # only search from new pixels
                        else:
                            peroxisomes[new_obj] = obj
                            tester = 1
                else:
                    self.on_edge[obj] = False
                    #TODO: FINISH IMPLEMENT GROWING PEROXISOMES OUTSIDE OF CELL!
        self.log.append('filtering out too-large and too-small objects...')

        obj_nums, volumes = np.unique(peroxisomes, return_counts = True)
        
        # the following code, in one line, eliminates all objects that are
        # either fewer than 5 or greater than 2500 pixels in volume.
        peroxisomes[np.in1d(peroxisomes, 
                            obj_nums[np.bitwise_or(volumes < 5,volumes >
                                                    2500)].astype('int')).reshape(peroxisomes.shape)]= 0
        obj_nums, volumes = np.unique(peroxisomes, return_counts = True)

        if self.mode == 'threshold':
            return PexSegmentObj(f_directory, self.filename, raw_img,
                                 gaussian_img, self.mode, 
                                 threshold_img, dist_map, smooth_dist, maxima,
                                 labs, peroxisomes, self.log,
                                 volumes, mode_params = self.threshold)
        elif self.mode == 'bg_scaled':
            return PexSegmentObj(f_directory, self.filename, raw_img,
                                 gaussian_img, self.mode, 
                                 threshold_img, dist_map, smooth_dist, maxima,
                                 labs, peroxisomes, self.log, volumes, 
                                 mode_params = [self.bg_diff,self.cells])

    ## HELPER METHODS ##
    def watershed_labels(self, maxima_img):
        '''Takes a boolean array with maxima labeled as true pixels
        and returns an array with maxima numbered sequentially.'''
        
        max_z, max_y, max_x = np.nonzero(maxima_img)
        
        label_output = np.zeros(maxima_img.shape)
        
        for i in range(0,len(max_y)):
            label_output[max_z[i],max_y[i],max_x[i]] = i+1
        
        return(label_output)
