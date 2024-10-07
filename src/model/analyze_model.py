from enumerations import GOAL, OBJECT_CLASSES, DETECTOR


class AnalyzeModel:
    def __init__(self, path_for_all_objects: str, detector_results_path: str,
                 goal=None, ):
        if goal is None:
            goal = GOAL.goal1
        if not isinstance(path_for_all_objects, str):
            raise ValueError(
                "Please provide a valid path")
        if not isinstance(detector_results_path, str):
            raise ValueError(
                "Please provide a valid path")
        self.goal = goal
        self.path_for_all_objects = path_for_all_objects
        self.detector_results_path = detector_results_path
