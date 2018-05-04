

### This code is associated with the paper from Weir et al., "The AAA protein Msp1 mediates clearance of excess tail-anchored proteins from the peroxisomal membrane". eLife, 2017. http://dx.doi.org/10.7554/eLife.28507


# pyto_segmenter

### developer: Nicholas Weir, Ph.D. Student, Denic Laboratory, Harvard University
### emai l: nweir a.t fas do t harvard (dot) edu


## Purpose:
This package was developed to identify cells, foci, and reticular structures from Z-stack fluorescence microscopy images (multipage TIFF format). It was developed during preparation of [Weir et al. 2017](https://doi.org/10.1101/136572).
__Please cite this manuscript if you use this package!__

## Installation:
As with any python module, clone the repository into your PYTHONPATH.

### Dependencies:
- python 3.x __Not compatible with Python 2__
- [matplotlib](https://matplotlib.org/)
- [scikit-image](http://scikit-image.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)

This in-development module is intended for segmenting _Saccharomyces cerevisiae_ cells (or other roughly elliptical cells) using the signal from a cytosolically-localized fluorescent protein in fluorescence microscopy z-stacks. It is not perfect, and segmented cells require pruning post-hoc to eliminate poorly segmented cells (~5-20% when imaged cells are present in a single layer, more otherwise).

## Contents:

### PexSegment.py
Classes and methods for segmenting small foci using a Canny algorithm (or using more basic thresholding methods if desired - see docstrings). Has methods to save intermediate images/output images as well as outputting segmented objects for further analysis.
### MitoSegment.py
Based on PexSegment.py, the classes provided here perform two additional steps:
- merging contiguous objects to generate longer reticular structures
- emptying holes in donut-shaped objects to avoid erroneous segmentation

### PytoSegment.py
Classes and methods for merging objects acquired in different channels (e.g. assign peroxisomes to "parent" cells). This code is still in active development.
### CellSegment.py
Classes and methods for identifying whole cells expressing a cytosolic marker fluorescent protein based on a Z-stack image. This was designed for roughly elliptical cells such as _Saccharomyces cerevisiae_. It does not do well when cells are not present in a single layer. See docstrings for additional details. This code is stil in active development.

## Usage examples
For usage examples, see the parallel repository for Weir et al. 2017 figure generation scripts in [The Denic Lab](https://github.com/deniclab)

Readme last updated 5.24.2017
