
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
        target = next(
            (label for label in objs if label.lower() in {"plate", "shelf"}),
            None,
        )
        if target is None:
            raise ValueError("The scene does not contain a plate or shelf target")

        plan = []
        for label in objs:
            if label == target:
                continue
            plan.extend(self.pick_place_object_plan(label, target, objs))
        plan.append({"action": "home"})
        return plan

    def pick_place_object_plan(self, obj_label, place_location_label, objs):
        '''
        returns a sequential action plan executable by manipulator to pick a object and place it at a specified location
        args
            obj_label : label of object to pick
            place_location label : label of object on which the object is to be placed
        '''
        pick_location, pick_orientation = self.perceptor.get_object_pick_pose(
            obj_label, objs
        )
        place_location, place_orientation = self.perceptor.get_obj_place_pose(
            obj_label, place_location_label, objs
        )
        return [
            {
                "action": "pick",
                "location": pick_location,
                "orientation": pick_orientation,
            },
            {"action": "home"},
            {
                "action": "place",
                "location": place_location,
                "orientation": place_orientation,
            },
            {"action": "home"},
        ]