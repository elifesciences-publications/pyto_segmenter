#### FUNCTIONS AND CLASSES FOR MICROSCOPY DIRECTORY MANIPULATION ####

def mk_fname_ref(img_list, wlength_string, delimiter = '_'):
    '''Create a reference dict for matching stage position images from
    different channels.
    
    Keyword arguments:
    img_list: the list of image files being worked on. contains all
    wavelengths.
    wlength_string: string containing identifying information for the channel
    of interest, e.g. '488' for images with filenames containing
    'w3.488.laser.25'.
    delimiter (optional): the delimiter used to break up the filename for
    removing the wavelength information.

    Returns: a dictionary whose key:value pairs are:
        key: the filename with the wavelength identifying fragment removed
        value: the full filename of the image
        for all images in the wavelength defined.
        Example of a key:value pair:
            a_stage_pos_fname.tf:a_stage_pos_wavelengthinfo_fname.tif
    '''

    wlength_imlist = [i for i in img_list if wlength_string in i]
    wlength_rm = []
    for fname in wlength_imlist:
        split_fname = fname.split(delimiter)
        fname_no_wlength = '_'.join([x for x in split_fname if wlength_string
                                     not in x])
        wlength_rm.append(fname_no_wlength)
    return dict(zip(wlength_rm, wlength_imlist))
