'''Classes and methods for segmentation of spherical objects within cells.'''


## IMPORT DEPENDENCIES
import matplotlib
matplotlib.use('Agg')
import os
import sys
import pickle
import time
from operator import itemgetter
import numpy as np
import pandas as pd
from skimage import io
from skimage.morphology import watershed
from skimage.feature import canny
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_closing
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.ndimage.morphology import binary_fill_holes, binary_opening
from scipy.ndimage import generic_gradient_magnitude, sobel
import matplotlib.pyplot as plt


class PexSegmentObj:
    '''An object containing information for segmented peroxisomes.'''
    def __init__(self, f_directory, filename, raw_img, gaussian_img, 
                 seg_method, mode, threshold_img, dist_map,
                 smooth_dist_map, maxima, labs, watershed_output, 
                 obj_nums, volumes, to_pdout = [], 
                 mode_params = {}):
        '''Initialize the PexSegmentObj with segmentation data.'''
        print('creating PexSegmentObj...')
        self.f_directory = f_directory
        self.filename = os.path.basename(filename).lower()
        self.raw_img = raw_img.astype('uint16')
        self.gaussian_img = gaussian_img.astype('uint16')
        self.seg_method = seg_method
        self.mode = mode
        self.threshold_img = threshold_img.astype('uint16')
        self.dist_map = dist_map.astype('uint16')
        self.smooth_dist_map = smooth_dist_map.astype('uint16')
        self.maxima = maxima.astype('uint16')
        self.labs = labs.astype('uint16')
        self.peroxisomes = watershed_output.astype('uint16')
        self.slices = self.raw_img.shape[0]
        self.height = self.raw_img.shape[1]
        self.width = self.raw_img.shape[2]
        self.obj_nums = obj_nums
        self.npexs = len(self.obj_nums)
        self.volumes = volumes
        self.volumes_flag = 'pixels'
        self.pdout = []
        self.border_rm_flag = False
        for key in mode_params:
            if hasattr(self, key):
                raise AttributeError('Two copies of the attribute ' + key +
                                     'were provided to PexSegmentObj.__init__()')
            setattr(self, key, mode_params[key])
        if to_pdout != []:
            for x in to_pdout:
                self.pdout.append(x)


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
        self.plot_stack(self.peroxisomes)
        if display == True:
            plt.show()

    def output_all_images(self, output_dir = None):
        '''Write all images to a new subdirectory.
        
        Write all images associated with the PexSegmentObj to a new
        directory. Name that directory according to the filename of the initial
        image that the object was derived from. This new directory should be a
        subdirectory to the directory containing the original raw image.
        '''
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.tif')]
        if not os.path.isdir(output_dir):
            print('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        print('writing images...')
        io.imsave('raw_'+self.filename, self.raw_img)
        io.imsave('gaussian_'+self.filename, self.gaussian_img)
        io.imsave('threshold_'+self.filename, self.threshold_img)
        io.imsave('dist_'+self.filename, self.dist_map)
        io.imsave('smooth_dist_'+self.filename,self.smooth_dist_map)
        io.imsave('maxima_'+self.filename,self.maxima)
        io.imsave('wshed_'+self.filename,self.peroxisomes)
        if hasattr(self,'edges'):
            io.imsave('edges_'+self.filename,self.edges)

    def output_image(self, imageattr, output_dir = None):
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.tif')]
        if not os.path.isdir(output_dir):
            print('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        print('writing image' + str(imageattr))
        io.imsave(str(imageattr)+self.filename, getattr(self,str(imageattr)))

    def output_plots(self):
        '''Write PDFs of slice-by-slice plots.
        
        Output: PDF plots of each image within PexSegmentObj in a directory
        named for the original filename they were generated from. Plots are
        generated using the plot_stack method and plotting methods defined
        here.
        '''
        os.chdir(self.f_directory)
        if not os.path.isdir(self.f_directory + '/' +
                             self.filename[0:self.filename.index('.tif')]):
            print('creating output directory...')
            os.mkdir(self.f_directory + '/' +
                     self.filename[0:self.filename.index('.tif')])
        os.chdir(self.f_directory + '/' +
                 self.filename[0:self.filename.index('.tif')])
        print('saving plots...')
        self.plot_raw_img()
        plt.savefig('praw_'+self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_gaussian_img()
        plt.savefig('pgaussian_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_threshold_img()
        plt.savefig('pthreshold_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        plt.savefig('pdist_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_smooth_dist_map()
        plt.savefig('psmooth_dist_' + 
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_maxima()
        plt.savefig('pmaxima_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
        self.plot_watershed()
        plt.savefig('pwshed_' +
                    self.filename[0:self.filename.index('.tif')]+'.pdf')
    def pickle(self, output_dir = None, filename = None):
        '''pickle the CellSegmentObj for later loading.'''
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.tif')]
        if filename == None:
            filename = self.filename[0:self.filename.index('.tif')] + '.pickle'
        if not os.path.isdir(output_dir):
            print('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    def output_all(self):
        os.chdir(self.f_directory)
        if not os.path.isdir(self.f_directory + '/' +
                             self.filename[0:self.filename.index('.tif')]):
            os.mkdir(self.f_directory + '/' +
                     self.filename[0:self.filename.index('.tif')])
        os.chdir(self.f_directory + '/' +
                 self.filename[0:self.filename.index('.tif')])
        print('outputting all data...')
        self.output_plots()
        self.output_all_images()
        self.pickle()
        # TODO: UPDATE THIS METHOD TO INCLUDE PANDAS OUTPUT
    def output_slim(self):
        '''output the slimmed object.'''
        # TODO: IMPLEMENT THIS METHOD (WILL REQUIRE CREATING OUTPUT_SLIM_PLOTS
        # AND OUTPUT_IMAGES_SLIM AS WELL; OR, I COULD CREATE A SLIM ATTRIBUTE
        # AND ADJUST THESE METHODS TO CHANGE DEPENDING UPON THE STATE OF THE
        # SLIM ATTRIBUTE)
        pass
    def to_csv(self, output_dir = None):
        os.chdir(self.f_directory)
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.tif')]
        if not os.path.isdir(output_dir):
            print('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        for_csv = self.to_pandas()
        for_csv.to_csv(path_or_buf = output_dir + '/' +
                       self.filename[0:self.filename.index('.tif')]+ '.csv',
                       index = True, header = True)
        
    ## HELPER METHODS ##

    def rm_border_objs(self, border = 1, z = True):
        '''remove all objects that contact the edge of the 3D stack.

        args:
            border: the size of the border around the edge which a pixel from
               the object must contact to be removed.
            z: should objects that contact the z-axis edge be eliminated? if
               true, any object with a pixel in the top or bottom image of the
               stack is removed.

        output: alters the objects within the PexSegmentObj. removes objects
        from the peroxisomes image, the obj_nums, and all other variables with
        elements of obj_nums as keys in a dict (parents, volumes, etc)
        '''

        border_mask = np.full(shape = self.peroxisomes.shape, fill_value = True,
                              dtype = bool)
        if z == True:
            border_mask[border:-border,border:-border,border:-border] = False
        elif z == False:
            border_mask[:,border:-border,border:-border] = False
        objs_to_rm = np.unique(self.peroxisomes[border_mask])
        objs_to_rm = objs_to_rm[objs_to_rm != 0]
        for x in objs_to_rm:
            self.peroxisomes[self.peroxisomes == x] = 0
            self.obj_nums.remove(x)
            self.volumes.pop(x, None)
            if hasattr(self, "parent"):
                self.parent.pop(x, None)
        self.npexs = len(self.obj_nums)
        self.border_rm_flag = True


    def to_pandas(self):
        '''create a pandas DataFrame of tabulated numeric data.
        
        the pdout attribute indicates which variables to include in the
        DataFrame.
        '''
        df_dict = {}
        for attr in self.pdout:
            df_dict[str(attr)] = pd.Series(getattr(self, attr))
        if 'volumes' in self.pdout:
            vflag_out = dict(zip(self.obj_nums,
                                 [self.volumes_flag]*len(self.obj_nums)))
            df_dict['volumes_flag'] = pd.Series(vflag_out)
        return pd.DataFrame(df_dict)
    def convert_volumes(self, z = 0.2, x = 0.0675):
        '''convert volumes from units of pixels to metric units.

        args:
            z: the distance between slices in the z-stack in units of microns.
            x: the linear distance between adjacent pixels in each slice in
            units of microns. x is also used for y.

        output: converts self.volumes to units of femtoliters, and changes the
        self.volumes_flag to 'femtoliters'.
        '''
        conv_factor = z*x*x
        for key, val in self.volumes:
            self.volumes[key] = float(self.volumes[key])*conv_factor
        self.volumes_flag = 'femtoliters' 
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
    
    def __init__(self,filename, seg_method = 'threshold', mode = 'threshold', **kwargs):
        self.filename = filename
        self.seg_method = seg_method
        self.mode = mode
        for key, value in kwargs.items():
            setattr(self,key,value)
        if self.seg_method == 'canny':
            self.high_threshold = kwargs.get('high_threshold',1000)
            self.low_threshold = kwargs.get('low_threshold',500)
        if self.seg_method == 'threshold':
            if mode == 'threshold':
                self.threshold = kwargs.get('threshold',float('nan'))
                if np.isnan(self.threshold):
                    raise ValueError('A threshold argument must be provided to segment with a constant threshold.')
            if mode == 'bg_scaled':
                self.cells = kwargs.get('cells', '')
                self.bg_diff = float(kwargs.get('bg_diff',float('nan')))
                if self.cells == '':
                    raise ValueError('A CellSegmentObj containing segmented cells is required if mode == bg_scaled.')
                if np.isnan(self.bg_diff):
                    raise ValueError('a bg_diff argument is needed if mode == bg_scaled.')
    def segment(self):
        '''Segment peroxisomes within the image.'''
        starttime = time.time() # begin timing
        f_directory = os.getcwd()
        pdout = []
        # data import
        print('reading' + self.filename)
        raw_img = io.imread(self.filename)
        print('raw image imported.')
        # gaussian filter assuming 100x objective and 0.2 um slices
        print('performing gaussian filtering...')
        gaussian_img = gaussian_filter(raw_img, [1,1,1])
        print('cytosolic image smoothed.')
        print('preprocessing complete.')
        ## SEGMENTATION BY THRESHOLDING THE GAUSSIAN ##
        if self.seg_method == 'threshold':
            # binary thresholding and cleanup
            print('thresholding...')
            threshold_img = np.copy(gaussian_img)
            if self.mode == 'threshold':
               print('mode = threshold.')
               threshold_img[threshold_img < self.threshold] = 0
               threshold_img[threshold_img > 0] = 1
               print('thresholding complete.')
            if self.mode == 'bg_scaled':
                print('mode = background-scaled.')
                self.thresholds = {}
                threshold_img = np.zeros(shape = raw_img.shape)
                for i in self.cells.obj_nums:
                    if i == 0:
                        pass
                    else:
                        print('thresholding cell ' + str(i))
                        cell_median = np.median(gaussian_img[self.cells.final_cells == i])
                        threshold_img[np.logical_and(self.cells.final_cells == i,
                                      gaussian_img > cell_median + self.bg_diff)] = 1
                        self.thresholds[i] = cell_median + self.bg_diff #store val
                print('thresholding complete.')
            # distance and maxima transformation to find objects
            # next two steps assume 100x objective and 0.2 um slices
            print('generating distance map...')
            dist_map = distance_transform_edt(threshold_img, sampling = (2,1,1))
            print('distance map complete.')
            print('smoothing distance map...')
            smooth_dist = gaussian_filter(dist_map, [1,2,2])
            print('distance map smoothed.')
            print('identifying maxima...')
            max_strel = generate_binary_structure(3,2)
            maxima = maximum_filter(smooth_dist,
                                    footprint = max_strel) == smooth_dist
            # clean up background and edges
            bgrd_3d = smooth_dist == 0
            eroded_bgrd = binary_erosion(bgrd_3d, structure = max_strel,
                                         border_value = 1)
            maxima = np.logical_xor(maxima, eroded_bgrd)
            print('maxima identified.')
            # watershed segmentation
            labs = self.watershed_labels(maxima)
            print('watershedding...')
            peroxisomes = watershed(-smooth_dist, labs, mask = threshold_img)
            print('watershedding complete.')
            if self.mode == 'bg_scaled':
                edge_struct = generate_binary_structure(3,1)
                self.c_edges = {}
                print('finding edges of cells...')
                for i in self.cells.obj_nums:
                    self.c_edges[i] = np.logical_xor(self.cells.final_cells == i,
                                                          binary_erosion(self.cells.final_cells== i,
                                                                         edge_struct))
                print('cell edges found.')
                self.primary_objs = [x for x in np.unique(peroxisomes) if x != 0]
                self.parent = {}
                self.obj_edges = {}
                self.on_edge = {}
                pex_mask = peroxisomes != 0
                for obj in self.primary_objs:
                    self.parent[obj] = self.cells.final_cells[labs == obj][0]
                    obj_mask = peroxisomes == obj
                    obj_edge = np.logical_xor(obj_mask, 
                                              binary_erosion(obj_mask,
                                                             edge_struct))
                    self.obj_edges[obj] = obj_edge
                    # test if the object's edge and its cell's edge overlap
                    if np.any(np.logical_and(obj_edge,
                                             self.c_edges[self.parent[obj]])):
                        self.on_edge[obj] = True
                        print('object on the edge: ' + str(obj))
                        print('parent cell: ' + str(self.parent[obj]))
                        new_obj = obj_mask
                        search_obj = obj_mask
                        tester = 0
                        iteration = 1
                        while tester == 0:
                            # TODO: FIX THIS BLOCK OF CODE! GETTING STUCK WITHIN
                            # IT! NOT SURE HOW MANY ITERATIONS ITS DOING, OR FOR
                            # HOW MANY DIFFERENT PEROXISOMES.
                            new_px = binary_dilation(search_obj, edge_struct)
                            new_px[np.logical_or(new_obj, pex_mask)] = False
                            print('iteration: ' + str(iteration))
                            # print('new pixels for iteration ' + str(iteration) + \
                            #      ': ')
                            # print(np.nonzero(new_px))
                            if np.any(gaussian_img[new_px] >
                                      self.thresholds[self.parent[obj]]):
                                to_add = np.logical_and(new_px, gaussian_img >
                                                        self.thresholds[self.parent[obj]])
                                new_obj = np.logical_or(new_obj, to_add)
                            #    print('object pixels after iteration '
                            #          + str(iteration) + ': ')
                            #    print(np.nonzero(new_obj))
                                search_obj = to_add # only search from new pixels
                            else:
                                peroxisomes[new_obj] = obj
                                tester = 1
                            iteration = iteration + 1
                    else:
                        self.on_edge[obj] = False
        elif self.seg_method == 'canny':
            ## EDGE-DETECTION BASED SEGMENTATION ##
            threshold_img = np.empty_like(gaussian_img)
            edge_img = np.empty_like(gaussian_img)
            c_strel = generate_binary_structure(2,1)
            for s in range(0,gaussian_img.shape[0]):
                c = canny(gaussian_img[s,:,:],
                          sigma = 0,
                          low_threshold = self.low_threshold,
                          high_threshold = self.high_threshold)
                c = binary_closing(c,c_strel)
                edge_img[s,:,:] = np.copy(c)
                c = binary_fill_holes(c)
                c = binary_opening(c, c_strel) # eliminate incomplete lines
                threshold_img[s,:,:] = c
            dist_map = distance_transform_edt(threshold_img, sampling = (3,1,1))
            print('distance map complete.')
            print('smoothing distance map...')
            smooth_dist = gaussian_filter(dist_map, [1,2,2])
            print('distance map smoothed.')
            print('identifying maxima...')
            max_strel = generate_binary_structure(3,2)
            maxima = maximum_filter(smooth_dist,
                                    footprint = max_strel) == smooth_dist
            # clean up background and edges
            bgrd_3d = smooth_dist == 0
            eroded_bgrd = binary_erosion(bgrd_3d, structure = max_strel,
                                         border_value = 1)
            maxima = np.logical_xor(maxima, eroded_bgrd)
            print('maxima identified.')
            # watershed segmentation
            labs = self.watershed_labels(maxima)
            print('watershedding...')
            peroxisomes = watershed(-smooth_dist, labs, mask = threshold_img)
            print('watershedding complete.')
            if hasattr(self,'cells'):
                self.primary_objs = [x for x in np.unique(peroxisomes) \
                                     if x != 0]
                self.parent = {}
                for obj in self.primary_objs:
                    o_parent = self.cells.final_cells[labs == obj][0]
                    if o_parent == 0:
                        self.primary_objs.remove(obj)
                    else:
                        self.parent[obj] = o_parent
        for s in range(1,peroxisomes.shape[0]):
            cslice = peroxisomes[s,:,:]
            lslice = peroxisomes[s-1,:,:]
            for obj in np.unique(cslice)[np.unique(cslice)!= 0]:
                lslice_vals, cts = np.unique(lslice[cslice == obj],
                                             return_counts = True)
                lslice_vals = lslice_vals.tolist()
                cts = cts.tolist()
                ordered_by_ct = sorted(zip(lslice_vals, cts),
                                       key = itemgetter(1))
                if ordered_by_ct[-1][0] == 0 or ordered_by_ct[-1][0] == obj:
                    continue
                else:
                    # if >75% of pixels in the slice below obj are from another
                    # object, change obj to that object #
                    if float(ordered_by_ct[-1][1])/cslice[cslice == obj].size>0.5:
                        peroxisomes[s,:,:][cslice == obj] = ordered_by_ct[-1][0]
        print('filtering out too-large and too-small objects...')
        obj_nums, volumes = np.unique(peroxisomes, return_counts = True)
        volumes = dict(zip(obj_nums.astype('uint16'), volumes))
        del volumes[0]
        obj_nums = obj_nums.astype('uint16').tolist()
        obj_nums.remove(0)
        for obj in obj_nums:
            if volumes[obj] > 3000:
                # delete the object AND exclude its parent cell from analysis
                if hasattr(self, 'cells'):
                    self.cells.obj_nums.remove(self.parent[obj])
                    self.cells.final_cells[self.cells_final_cells == 
                                           self.parent[obj]] = 0
                    del volumes[obj]
                    obj_nums.remove(obj)
        # merge objects that segmented into diff objects by Z slice
        mode_params = {}
        if hasattr(self, 'parent'):
            pdout.append('parent')
            mode_params['parent'] = self.parent
        if self.seg_method == 'canny':
            mode_params['high_threshold'] = self.high_threshold
            mode_params['low_threshold'] = self.low_threshold
            mode_params['edges'] = edge_img
            pdout.append('volumes')
        if self.seg_method == 'threshold':
            if self.mode == 'threshold':
                mode_params['threshold'] = self.threshold
                pdout.append('volumes')
            elif self.mode == 'bg_scaled':
                mode_params['thresholds'] = self.thresholds
                mode_params['bg_diff'] = self.bg_diff
                mode_params['cells'] = self.cells
                mode_params['cell_edges'] = self.c_edges
                mode_params['cell_nums'] = self.cells.obj_nums
                mode_params['obj_edges'] = self.obj_edges
                mode_params['on_edge'] = self.on_edge
                for x in ['thresholds','on_edge','parent', 'volumes']:
                    pdout.append(x)
        return PexSegmentObj(f_directory, self.filename, raw_img,
                             gaussian_img, self.seg_method, self.mode, 
                             threshold_img, dist_map, smooth_dist, maxima,
                             labs, peroxisomes, obj_nums, volumes,
                             to_pdout = pdout, mode_params = mode_params)

    ## HELPER METHODS ##
    def watershed_labels(self, maxima_img):
        '''Takes a boolean array with maxima labeled as true pixels
        and returns an array with maxima numbered sequentially.'''
        
        max_z, max_y, max_x = np.nonzero(maxima_img)
        
        label_output = np.zeros(maxima_img.shape)
        
        for i in range(0,len(max_y)):
            label_output[max_z[i],max_y[i],max_x[i]] = i+1
        
        return(label_output)

    


# TODO LIST:
    # # UPDATE LOG FILE PRINTING
