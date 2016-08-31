# pyto_segmenter

Readme last updated 5.1.2016

## NOTE: 
###THIS PROJECT IS IN ACTIVE DEVELOPMENT AND IS NOT CURRENTLY PREPARED FOR DISTRIBUTION. IF YOU'RE INTERESTED IN USING THE CODE IN THIS PROJECT IN THE SHORT TERM, I RECOMMEND CONTACTING ME DIRECTLY.

This in-development module is intended for segmenting _Saccharomyces cerevisiae_ cells (or other roughly elliptical cells) using the signal from a cytosolically-localized fluorescent protein in fluorescence microscopy z-stacks. It is not perfect, and segmented cells require pruning post-hoc to eliminate poorly segmented cells (~5-20% when imaged cells are present in a single layer, more otherwise).

##Contents:
###CellSegment.py
Classes and methods for identifying whole cells expressing a cytosolic marker, along with saving intermediates produced during the process and the segmented output.
###PexSegment.py
Classes and methods for segmenting small foci using a Canny algorithm (or using more basic thresholding methods if desired - see code). Has methods to save intermediate images/output images as well as outputting segmented objects for further analysis.
###PytoSegment.py
Classes and methods for merging objects acquired in different channels (e.g. assign peroxisomes to "parent" cells).

developer: Nicholas Weir, Ph.D. Student, Denic Laboratory, Harvard University
emai l: nweir a.t fas do t harvard (dot) edu

_Please cite this work if you use it!_ 

[![DOI](https://zenodo.org/badge/16661/nrweir/pyto_segmenter.svg)](https://zenodo.org/badge/latestdoi/16661/nrweir/pyto_segmenter)

