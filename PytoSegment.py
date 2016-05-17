'''Classes for merging segmented cells with subcellular structures'''


class PytoSegmentObj:
    '''A superclass to merge together a CellSegmentObj with child objects
    segmented in other channels.
    '''

    def __init__(self, cell_obj, daughter_objs = {}):
        '''Generate a PytoSegmentObj instance.'''
        self.cells = cell_obj
        for key in daughter_objs:
            setattr(self, key, daughter_objs[key])
    def match_parents(self, obj_type):
        '''Match subcellular objects to their parent cell.

        args:
            obj_type: the string used when creating the PytoSegmentObj to
                      name the object (e.g. peroxisomes)
        '''
        temp_objs = getattr(self, obj_type)
        if not hasattr(temp_objs, labs):
            raise AttributeError('The ' + obj_type +
                                 ' lacks a labs attribute.')
        parents = {}
        labs = temp_objs.labs
        for obj in temp_objs.obj_nums:
            parents[obj] = self.cells.final_cells[labs == obj]
        temp_objs.parents = parents
        setattr(self, obj_type, temp_objs)
