'''Classes for merging segmented cells with subcellular structures'''

import pandas as pd
import pickle

class PytoSegmentObj:
    '''A metaclass to merge together a CellSegmentObj with child objects
    segmented in other channels.
    '''

    def __init__(self, cell_obj, daughter_objs = {}):
        '''Generate a PytoSegmentObj instance.'''
        self.cells = cell_obj
        for key in daughter_objs:
            setattr(self, key, daughter_objs[key])


    ## OUTPUT METHODS ##

    def pickle(self):
        '''pickle the PytoSegmentObj for later loading.'''
        if not os.path.isdir(self.cells.f_directory + '/' +
                             self.cells.filename[0:self.cells.filename.index('.')]):
            self.log.append('creating output directory...')
            os.mkdir(self.cells.f_directory + '/' + 
                     self.cells.filename[0:self.cells.filename.index('.')])
        os.chdir(self.cells.f_directory + '/' + 
                 self.cells.filename[0:self.cells.filename.index('.')])
        with open('pickled_PytoSegmentObj_' +
                  self.cells.filename[0:self.cells.filename.index('.')] + 
                  '.pickle', 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()


    ## HELPER METHODS ##

    def match_parents(self, obj_type):
        '''Match subcellular objects to their parent cell.

        args:
            obj_type: the string used when creating the PytoSegmentObj to
                      name the object (e.g. peroxisomes)
        '''
        tmp_objs = getattr(self, obj_type)
        if not hasattr(tmp_objs, 'labs'):
            raise AttributeError('The ' + obj_type +
                                 ' lacks a labs attribute.')
        parent = {}
        labs = tmp_objs.labs
        for obj in tmp_objs.obj_nums:
            parent[obj] = self.cells.final_cells[labs == obj]
        tmp_objs.parent = parent
        setattr(self, obj_type, tmp_objs)
    def count_children(self, obj_type):
        '''Count child objects of a given type in each cell.

        args:
            obj_type: the object class to be classified as a child within the
            cells.

        output: adds an attribute to self.cells named str(obj_type) +
        '_children' which is a dict of cell number:number of child objs pairs.
        '''
        tmp_objs = getattr(self, obj_type)
        if not hasattr(tmp_objs, 'parent'):
            self.match_parents(obj_type)
        cts = {}
        tmp_parents = [int(val) for val in tmp_objs.parent.values()]
        for cell in self.cells.obj_nums:
            cts[cell] = tmp_parents.count(int(cell))
        setattr(self.cells, str(obj_type) + '_children', cts)
        # add this children attribute to pandas-formatted output
        self.cells.pdout.append(str(obj_type) + '_children')
    
