from enumerations import GOAL, OBJECT_CLASSES, DETECTOR


class AnalyzeModel:
    def __init__(self, path_for_all_objects: {},
                 goal=None, ):
        if goal is None:
            goal = GOAL.goal1
        if not isinstance(path_for_all_objects, str):
            raise ValueError(
                "Please provide a valid path")
        self.goal = goal
        self.path_for_all_objects = path_for_all_objects
