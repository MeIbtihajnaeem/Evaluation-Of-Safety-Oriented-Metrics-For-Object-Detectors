from src.controller.analyze_results_controller import AnalyzeResultsController
from src.model.analyze_model import AnalyzeModel
from src.utils.class_implementations.analyzer_utils import AnalyzerUtils
from enumerations import GOAL


def compute_example():
    goal = GOAL.goal1
    detector_results_path = "/Users/ibtihajnaeem/Documents/version_control/thesis/detectAndTrajectoryPackage/assets/mmdetection3d/GOAL1/retry_allobjects/POINTP/"
    path_for_all_objects = "/Users/ibtihajnaeem/Documents/version_control/thesis/detectAndTrajectoryPackage/assets/mmdetection3d/GOAL1/all_objects/"
    analyze_model = AnalyzeModel(goal=goal, path_for_all_objects=path_for_all_objects,
                                 detector_results_path=detector_results_path)
    analyzer_utils = AnalyzerUtils()
    obj = AnalyzeResultsController(analyzeModel=analyze_model, analyzerUtils=analyzer_utils)
    obj.compute()
compute_example()
