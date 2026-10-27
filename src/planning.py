
from perception import Perception
import numpy as np




class Planner:
    def __init__(self, manipulator):
        self.perceptor = Perception(manipulator)

    def get_clear_task_plan(self, objs):
        '''
        returns a sequential action plan executable by manipulator to place all the objects in the plate
        you can write the plans for specific objects such that plan steps are manipulator function calls, and robust to object location randomization
        '''
        # TODO

    def pick_place_object_plan(self, obj_label, place_location_label, objs):
        '''
        returns a sequential action plan executable by manipulator to pick a object and place it at a specified location
        args
            obj_label : label of object to pick
            place_location label : label of object on which the object is to be placed
        '''
        # TODO