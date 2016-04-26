## IMPORT DEPENDENCIES
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.morphology import watershed
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion 
from scipy.ndimage.morphology import binary_dilation, binary_fill_holes
from scipy.ndimage.morphology import distance_transform_edt


class PytoSegmentObj:
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
            segmented image using a voting filter. see the PytoSegmenter
            reassign_pix_obs method for details. 
    '''

    def __init__(self, f_directory, filename, raw_img, gaussian_img, threshold,
                 threshold_img, filled_img, dist_map, smooth_dist_map,
                 maxima, labs, watershed_output, filled_cells, final_cells,
                 segmentation_log):
        '''initialize the PytoSegmentObject with segmentation data.'''
        self.f_directory = f_directory
        self.filename = filename
        self.raw_img = raw_img
        self.gaussian_img = gaussian_img
        self.threshold = threshold
        self.threshold_img = threshold_img
        self.filled_img = filled_img
        self.dist_map = dist_map
        self.smooth_dist_map = smooth_dist_map
        self.maxima = maxima
        self.labs = labs
        self.watershed_output = watershed_output
        self.filled_cells = filled_cells
        self.final_cells = final_cells
        self.slices = self.raw_img.shape[0]
        self.height = self.raw_img.shape[1]
        self.width = self.raw_img.shape[2]

    def __repr__(self):
        return 'PytoSegmentObj '+ self.get_filename()

    ## PLOTTING METHODS ##    
    def plot_raw_img(self):
        plot_stack(self.raw_img, colormap = 'gray')
    def plot_gaussian_img(self):
        plot_stack(self.gaussian_img, colormap = 'gray')
    def plot_threshold_img(self):
        plot_stack(self.threshold_img, colormap = 'gray')
    def plot_filled_img(self):
        plot_stack(self.filled_img, colormap = 'gray')
    def plot_dist_map(self):
        plot_stack(self.dist_map)
    def plot_smooth_dist_map(self):
        plot_stack(self.smooth_dist_map)
    def plot_maxima(self):
        vis_maxima = binary_dilation(self.maxima,
                                     structure = np.ones(shape = (1,5,5)))
        masked_maxima = np.ma.masked_where(vis_maxima == 0, vis_maxima)
        plot_masked_maxima(masked_maxima, self.smooth_dist_map)

    ## OUTPUT METHODS ##

    def output_images(self):
        '''Write all images to a new subdirectory.
        
        Write all images associated with the PytoSegmentObj to a new
        directory. Name that directory according to the filename of the initial
        image that the object was derived from. This new directory should be a
        subdirectory to the directory containing the original raw image.
        '''
        #TODO: Complete this method
        os.chdir(f_directory)
        if not os.path.isdir(f_directory + filename[0:filename.index('.')-1]):
            os.mkdir(f_directory + filename[0:filename.index('.')-1])
        os.chdir(f_directory + filename[0:filename.index('.')-1])
        io.imwrite('raw_'+filename, raw_img)
        io.imwrite('gaussian_'+filename, gaussian_img)
        io.imwrite('threshold_'+filename, threshold_img)
        io.imwrite('filled_threshold_'+filename, filled_img)
        io.imwrite('dist_'+filename, dist_map)
        io.imwrite('smooth_dist_'+filename,smooth_dist_map)
        io.imwrite('maxima_'+filename,maxima)
        io.imwrite('wshed_'+filename,watershed_output)
        io.imwrite('filled_cells_'+filename, filled_cells)
        io.imwrite('final_cells_'+filename, final_cells)
    def output_plots(self):
        '''Write PDFs of slice-by-slice plots.
        
        Output: PDF plots of each image within PytoSegmentObj in a directory
        named for the original filename they were generated from. Plots are
        generated using the plot_stack method and plotting methods defined
        here.
        '''

    ## HELPER METHODS ##

    def plot_stack(stack_arr, colormap='jet'):
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


    def plot_maxima_stack(masked_max, smooth_dist):
        ''' Creates a matplotlib plot object in which each slice from the image
        is displayed as a single subplot, in a 4-by-n matrix (n depends upon
        the number of slices in the image)'''

        nimgs = stack_arr.shape[0] # z axis of array dictates number of slices
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


class PytoSegmenter:
    
    def __init__(self, image):
        self.log = []
        segment(image)

    def segment(self, filename, threshold):
        ## start timing
        starttime = time.time()
        ## DATA IMPORT AND PREPROCESSING
        f_directory = os.getcwd()
        self.log.append('reading ' + filename + ' ...')
        raw_img = io.imread(filename)
        self.log.append('raw image imported.')
        # next step's gaussian filter assumes 100x obj and 0.2 um slices
        self.log.append('performing gaussian filtering...')
        gaussian_img = gaussian_filter(raw_img, [1,2,2])
        self.log.append('cytosolic image smoothed.')
        self.log.append('preprocessing complete.')
        ## BINARY THRESHOLDING AND IMAGE CLEANUP
        self.log.append('thresholding...')
        threshold_img = np.copy(gaussian_img)
        threshold_img[threshold_img < threshold] = 0
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
        labs = watershed_labels(maxima)
        self.log.append('watershedding...')
        cells = watershed(-smooth_dist,labs,mask = filled_img)
        self.log.append('raw watershedding complete.')
        self.log.append('filling 2d holes in cells...')
        filled_cells = fill_cells_2d(cells)
        self.log.append('2d hole-filling complete.')
        self.log.append('cleaning up cells...')
        clean_cells = reassign_pixels_3d(filled_cells)
        self.log.append('cell cleanup complete.')
        self.log.append('SEGMENTATION OPERATION COMPLETE.')
        endtime = time.time()
        runningtime = endtime - starttime
        self.log.append('time elapsed: ' + str(runningtime) + ' seconds')
        return PytoSegmentObj(f_directory, filename, raw_img, gaussian_img, threshold,
                 threshold_img, filled_img, dist_map, smooth_dist_map,
                 maxima, labs, watershed_output, filled_cells, final_cells,
                 segmentation_log):


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
        segmented image to clean up cells.'''
        
        reassigned_cells = np.ndarray(shape = filled_cells.shape)
        for i in range(0,filled_cells.shape[0]):
            self.log.append('    beginning slice ' + str(i) + '...')
            reassigned_cells[i,:,:] = reassign_pixels_2d(filled_cells[i,:,:])
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


class BatchPytoSegmenter:
    '''Batch segment all files in a directory.

    Attributes:
        directory: path to the files to be segmented.
        files: a dictionary 


