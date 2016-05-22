'''Classes and methods for segmenting cells based on cytosolic fluorescence.'''

## IMPORT DEPENDENCIES

import matplotlib
matplotlib.use('Agg')
import os
import sys
import pickle
import time
import pandas as pd
import numpy as np
from skimage import io
from skimage.morphology import watershed
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion 
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from scipy.ndimage.morphology import distance_transform_edt
import matplotlib.pyplot as plt



class CellSegmentObj:
    ''' segmentation data from a cytosolic fluorescence image.   
    Attributes:
        filename (str): the filename for the original raw fluroescence
            image.
        raw_img (ndarray): 3d array containing pixel intensities from
            the raw fluroescence image.
        slices: the number of z-slices in the raw image.
        height: the height of the raw image.
        width: the width of the raw image.
        gaussian_img (ndarray): gaussian-filtered fluorescence image.
        threshold (int): the cutoff used for binary thresholding of
            the gaussian image.
        threshold_img (ndarray): the output from applying the threshold
            to the gaussian image ndarray.
        filled_img (ndarray): the product of using three-dimensional
            binary hole filling (as implemented in
            scipy.ndimage.morphology) on the thresholded image
        dist_map (ndarray): a 3D euclidean distance transform of the
            filled image.
        smooth_dist_map (ndarray): gaussian smoothed distance map.
        maxima (ndarray): a boolean array indicating the positions of
            local maxima in the smoothed distance map.
        labs (ndarray): an int array with each local maximum in maxima
            assigned a unique integer value, and non-maxima assigned 0.
            required for watershed implementation at the next step.
        watershed_output (ndarray): a 3D array in which each object is
            assigned a unique integer value.
        filled_cells (ndarray): the product of performing 2D hole-filling
            object-by-object on the watershed_output array (holes are only
            filled if they are completely surrounded by the same cell;
            done to eliminate "tunnels" that can arise from vacuoles).
        final_cells (ndarray): the product of cleaning up the filled_cells
            segmented image using a voting filter. see the CellSegmenter
            reassign_pix_obs method for details. 
    '''

    def __init__(self, f_directory, filename, raw_img, gaussian_img, threshold,
                 threshold_img, filled_img, dist_map, smooth_dist_map,
                 maxima, labs, watershed_output, filled_cells, final_cells,
                 obj_nums, volumes, segmentation_log):
        '''initialize the CellSegmentObject with segmentation data.'''
        self.log = segmentation_log
        self.log.append('creating CellSegmentObject...')
        self.f_directory = f_directory
        self.filename = os.path.basename(filename)
        self.raw_img = raw_img.astype('uint16')
        self.gaussian_img = gaussian_img.astype('uint16')
        self.threshold = int(threshold)
        self.threshold_img = threshold_img.astype('uint16')
        self.filled_img = filled_img.astype('uint16')
        self.dist_map = dist_map.astype('uint16')
        self.smooth_dist_map = smooth_dist_map.astype('uint16')
        self.maxima = maxima.astype('uint16')
        self.labs = labs.astype('uint16')
        self.watershed_output = watershed_output.astype('uint16')
        self.filled_cells = filled_cells.astype('uint16')
        self.final_cells = final_cells.astype('uint16')
        self.slices = int(self.raw_img.shape[0])
        self.height = int(self.raw_img.shape[1])
        self.width = int(self.raw_img.shape[2])
        self.obj_nums = obj_nums
        self.volumes = volumes
        self.volumes_flag = 'pixels'
        self.pdout = ['volumes']

    def __repr__(self):
        return 'CellSegmentObj '+ self.filename

    ## PLOTTING METHODS ##    
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
    def plot_filled_img(self, display = False):
        self.plot_stack(self.filled_img, colormap = 'gray')
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
    def plot_filled_cells(self, display = False):
        self.plot_stack(self.filled_cells)
        if display == True:
            plt.show()
    def plot_final_cells(self, display = False):
        self.plot_stack(self.final_cells)
        if display == True:
            plt.show()

    ## OUTPUT METHODS ##
    
    def to_csv(self, output_dir = None):
        os.chdir(self.f_directory)
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.')]
        if not os.path.isdir(output_dir):
            self.log.append('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        for_csv = self.to_pandas()
        for_csv.to_csv(path = output_dir + self.filename[0:self.filename.index('.')],
                       index = True, header = True)
    def output_image(self, imageattr, output_dir = None):
        os.chdir(self.f_directory)
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.')]
        if not os.path.isdir(output_dir):
            self.log.append('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        self.log.append('writing image' + str(imageattr))
        io.imsave(str(imageattr)+self.filename, getattr(self,str(imageattr)))

    def output_all_images(self, output_dir = None):
        '''Write all images to a new subdirectory.
        
        Write all images associated with the CellSegmentObj to a new
        directory. Name that directory according to the filename of the initial
        image that the object was derived from. This new directory should be a
        subdirectory to the directory containing the original raw image.
        '''
        os.chdir(self.f_directory)
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.')]
        if not os.path.isdir(output_dir):
            self.log.append('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
        self.log.append('writing images...')
        io.imsave('raw_'+self.filename, self.raw_img)
        io.imsave('gaussian_'+self.filename, self.gaussian_img)
        io.imsave('threshold_'+self.filename, self.threshold_img)
        io.imsave('filled_threshold_'+self.filename, self.filled_img)
        io.imsave('dist_'+self.filename, self.dist_map)
        io.imsave('smooth_dist_'+self.filename,self.smooth_dist_map)
        io.imsave('maxima_'+self.filename,self.maxima)
        io.imsave('wshed_'+self.filename,self.watershed_output)
        io.imsave('filled_cells_'+self.filename, self.filled_cells)
        io.imsave('final_cells_'+self.filename, self.final_cells)
    def output_plots(self):
        '''Write PDFs of slice-by-slice plots.
        
        Output: PDF plots of each image within CellSegmentObj in a directory
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
        self.plot_filled_img()
        plt.savefig('pfilled_' +
                    self.filename[0:self.filename.index('.')]+'.pdf')
        self.plot_dist_map()
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
        self.plot_filled_cells()
        plt.savefig('pfilled_cells_' +
                    self.filename[0:self.filename.index('.')]+'.pdf')
        self.plot_final_cells()
        plt.savefig('pfinal_cells_' +
                    self.filename[0:self.filename.index('.')]+'.pdf')
    def pickle(self, output_dir = None):
        '''pickle the CellSegmentObj for later loading.'''
        if output_dir == None:
            output_dir = self.f_directory + '/' + self.filename[0:self.filename.index('.')]
        if not os.path.isdir(output_dir):
            self.log.append('creating output directory...')
            os.mkdir(output_dir)
        os.chdir(output_dir)
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
        self.output_all_images()
        self.mk_log_file('log_'+self.filename[0:self.filename.index('.')]+'.txt')
        self.pickle()
         # TODO: UPDATE THIS METHOD TO INCLUDE PANDAS OUTPUT           
    ## HELPER METHODS ##

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
                                 self.volumes_flag*len(self.obj_nums)))
            df_dict['volumes_flag'] = pd.Series(vflag_out)
        return pd.DataFrame(df_dict)
    def convert_volumes(self, z = 0.2, x = 0.0675):
        '''convert volumes from units of pixels to metric units.

        args:
            z: the distance between slices in the z-stack in units of microns.
            defaults to commonly used setting for Murray 100x objective.
            x: the linear distance between adjacent pixels in each slice in
            units of microns. x is also used for y. Defaults to appropriate
            value for images acquired on the Murray 100X TIRF objective.

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
            f.show() # TODO: IMPLEMENT OPTIONAL SAVING OF THE PLOT

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
            f.show() # TODO: IMPLEMENT OPTIONAL SAVING OF THE PLOT
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
            f.show() # TODO: IMPLEMENT OPTIONAL SAVING OF THE PLOT

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
            f.show() # TODO: IMPLEMENT OPTIONAL SAVING OF THE PLOT
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
            f.write('binary threshold: ' + str(self.threshold) + '\n')
            f.close()

    ## RESTRUCTURING METHODS ##
    def slim(self):
        '''remove all of the processing intermediates from the object, leaving
        only the core information required for later analysis. primarily
        intended for use when doing batch analysis of multiple images, and
        combining CellSegmentObj instances with instances of other types of
        objects segmented in a different fluorescence channel.
        '''
        del self.raw_img
        del self.gaussian_img
        del self.threshold_img
        del self.filled_img
        del self.dist_map
        del self.smooth_dist_map
        del self.maxima
        del self.labs
        del self.watershed_output
        del self.filled_cells

        return self

class CellSegmenter:
    
    def __init__(self, filename, threshold):
        self.log = []
        self.filename = filename
        self.threshold = threshold

    def segment(self):
        ## start timing
        starttime = time.time()
        ## DATA IMPORT AND PREPROCESSING
        f_directory = os.getcwd()
        self.log.append('reading ' + self.filename + ' ...')
        raw_img = io.imread(self.filename)
        self.log.append('raw image imported.')
        # next step's gaussian filter assumes 100x obj and 0.2 um slices
        self.log.append('performing gaussian filtering...')
        gaussian_img = gaussian_filter(raw_img, [1,2,2])
        self.log.append('cytosolic image smoothed.')
        self.log.append('preprocessing complete.')
        ## BINARY THRESHOLDING AND IMAGE CLEANUP
        self.log.append('thresholding...')
        threshold_img = np.copy(gaussian_img)
        threshold_img[threshold_img < self.threshold] = 0
        threshold_img[threshold_img > 0] = 1
        self.log.append('thresholding complete.')
        self.log.append('filling holes...')
        filled_img = binary_fill_holes(threshold_img)
        self.log.append('3d holes filled.')
        self.log.append('binary processing complete.')
        ## DISTANCE AND MAXIMA TRANFORMATIONS TO FIND CELLS
        # next two steps assume 100x obj and 0.2 um slices
        self.log.append('generating distance map...')
        dist_map = distance_transform_edt(filled_img, sampling = (2,1,1))
        self.log.append('distance map complete.')
        self.log.append('smoothing distance map...')
        smooth_dist = gaussian_filter(dist_map, [2,4,4])
        self.log.append('distance map smoothed.')
        self.log.append('identifying maxima...')
        max_strel_3d = generate_binary_structure(3,2)
        maxima = maximum_filter(smooth_dist,
                                footprint = max_strel_3d) == smooth_dist
        # clean up background/edges
        bgrd_3d = smooth_dist == 0
        eroded_background_3d = binary_erosion(bgrd_3d, structure = max_strel_3d,
                                              border_value = 1)
        maxima = np.logical_xor(maxima, eroded_background_3d)
        self.log.append('maxima identified.')
        ## WATERSHED SEGMENTATION
        labs = self.watershed_labels(maxima)
        self.log.append('watershedding...')
        cells = watershed(-smooth_dist,labs,mask = filled_img)
        self.log.append('raw watershedding complete.')
        self.log.append('filling 2d holes in cells...')
        filled_cells = self.fill_cells_2d(cells)
        self.log.append('2d hole-filling complete.')
        self.log.append('cleaning up cells...')
        clean_cells = self.reassign_pixels_3d(filled_cells)
        self.log.append('cell cleanup complete.')
        self.log.append('SEGMENTATION OPERATION COMPLETE.')
        endtime = time.time()
        runningtime = endtime - starttime
        self.log.append('time elapsed: ' + str(runningtime) + ' seconds')
        cell_nums, volumes = np.unique(clean_cells, return_counts = True)
        cell_nums.astype('uint16')
        volumes.astype('uint16')
        volumes = dict(zip(cell_nums, volumes))
        del volumes[0]
        cell_nums = cell_nums[np.nonzero(cell_nums)]
        return CellSegmentObj(f_directory, self.filename, raw_img,
                              gaussian_img, self.threshold,
                              threshold_img, filled_img, dist_map,
                              smooth_dist, maxima, labs, cells, filled_cells, 
                              clean_cells, cell_nums, volumes, self.log)


    def watershed_labels(self, maxima_img):
        '''Takes a boolean array with maxima labeled as true pixels
        and returns an array with maxima numbered sequentially.'''
        
        max_z, max_y, max_x = np.nonzero(maxima_img)
        
        label_output = np.zeros(maxima_img.shape)
        
        for i in range(0,len(max_y)):
            label_output[max_z[i],max_y[i],max_x[i]] = i+1
        
        return(label_output)



    def fill_cells_2d(self, cell_img):
        '''Go cell-by-cell in a watershed-segmented image. 
        First, make a new 3D array that just includes that cell; then,
        fill holes within that cell in each 2D plane. Then add this cell
        back into a new 3D array with the same numbering scheme that the
        original watershedded image had.'''

        #initialize output image for adding cell data in as it's generated.
        filled_cells = np.zeros(shape = cell_img.shape)
        cell_img.astype(int)
        cells = np.unique(cell_img)
        ncells = len(cells)
        nplanes = cell_img.shape[0]

        for i in range(1,ncells):
            c_cell = cell_img == i
            for z in range(0,nplanes):
                c_cell[z,:,:] = binary_fill_holes(c_cell[z,:,:])
            filled_cells[c_cell == 1] = i
            self.log.append('cell ' + str(i) + ' of ' 
                            + str(ncells-1) + ' done.')
        return filled_cells

    def reassign_pixels_3d(self, filled_cells):
        '''Uses reassign_pixels_2d on each slice of a watershed-
        segmented image to clean up cells.
        '''

        reassigned_cells = np.ndarray(shape = filled_cells.shape)
        for i in range(0,filled_cells.shape[0]):
            self.log.append('    beginning slice ' + str(i) + '...')
            reassigned_cells[i,:,:] = self.reassign_pixels_2d(filled_cells[i,:,:])
            self.log.append('    slice ' + str(i) + ' complete.')
        return reassigned_cells

    def reassign_pixels_2d(self, watershed_img_2d):
        '''In 2D, go over each segmented pixel in the image with a 
        structuring element. Measure the frequency of each different 
        pixel value that occurs within this structuring element (i.e. 1 
        for one segmented cell, 2 for another, etc. etc.). Generate a 
        new image with each pixel value set to that of the most common 
        non-zero (non-background) pixel within its proximity. After 
        doing this for every pixel, check and see if the new image is
        identical to the old one. If it is, stop; if not, repeat with
        the new image as the starting image.'''

        #retrieve number of cells in the watershedded image
        ncells = np.unique(watershed_img_2d).size-1
        strel = np.array(
            [[0,0,0,1,0,0,0],
             [0,0,1,1,1,0,0],
             [0,1,1,1,1,1,0],
             [1,1,1,1,1,1,1],
             [0,1,1,1,1,1,0],
             [0,0,1,1,1,0,0],
             [0,0,0,1,0,0,0]])
        old_img = watershed_img_2d
        mask_y,mask_x = np.nonzero(old_img)
        tester = 0
        counter = 0
        while tester == 0:
            new_img = np.zeros(old_img.shape)
            for i in range(0,len(mask_y)):
                y = mask_y[i]
                x = mask_x[i]
                # shift submatrix if the pixel is too close to the edge to be 
                # centered within it
                if y-3 < 0:
                    y = 3
                if y+4 >= old_img.shape[0]:
                    y = old_img.shape[0] - 4
                if x-3 < 0:
                    x = 3
                if x+4 >= old_img.shape[1]:
                    x = old_img.shape[1]-4
                # take a strel-sized matrix around the pixel of interest.
                a = old_img[y-3:y+4,x-3:x+4]
                # mask for the pixels that I'm interested in comparing to
                svals = np.multiply(a,strel) 
                # convert this to a 1D array with zeros removed
                cellvals = svals.flatten()[np.nonzero(
                        svals.flatten())].astype(int) 
                # count number of pixels that correspond to each 
                # cell in proximity
                freq = np.bincount(cellvals,minlength=ncells)
                # find cell with most abundant pixels in vicinity
                top_cell = np.argmax(freq)
                # because argmax only returns the first index if there are
                # multiple matches, I need to check for duplicates. if 
                # there are duplicates, I'm going to leave the value as the 
                # original, regardless of whether the set of most abundant 
                # pixel values contain the original pixel (likely on an 
                # edge) or not (likely on a point where three cells come
                # together - this is far less common)    
                if np.sum(freq==freq[top_cell]) > 1:
                    new_img[mask_y[i],mask_x[i]] = old_img[mask_y[i],mask_x[i]]
                else:
                    new_img[mask_y[i],mask_x[i]] = top_cell

            if np.array_equal(new_img,old_img):
                tester = 1
            else:
                old_img = np.copy(new_img)
            counter += 1
        self.log.append('    number of iterations = ' + str(counter))
        return new_img


if __name__ == '__main__':
    ''' Run segmentation on all images in the working directory.

    sys.argv contents:
        threshold: a number that will be used as the threshold to binarize
        the cytosolic fluorescence image.
    '''

    threshold = int(sys.argv[1])
    wd_contents = os.listdir(os.getcwd())
    imlist = []
    for f in wd_contents:
        if (f.endswith('.tif') or f.endswith('.tiff') 
            or f.endswith('.TIF') or f.endswith('.TIFF')):
            imlist.append(f)
    self.log.append('final imlist:')
    self.log.append(imlist)
    for i in imlist:
        self.log.append('initializing segmenter object...')
        i_parser = CellSegmenter(i,threshold)
        self.log.append('initializing segmentation...')
        i_obj = i_parser.segment()
        i_obj.output_all()


# TODO LIST:
    # ADD LOG ATTRIBUTE APPEND COMMANDS IN METHODS WHERE IT'S NOT IMPLEMENTED
    # STOP THE PLOTS FROM BEING SHOWN WHEN THEY SHOULDN'T
