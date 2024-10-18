from src.controller.analyze_results_controller import AnalyzeResultsController
from model.DTOs.analyze_model import AnalyzeModel
from src.utils.class_implementations.analyzer_utils import AnalyzerUtils
from enumerations import GOAL


def compute_example_goal1():
    # SOLUTION FOR GOAL 1
    goal = GOAL.goal1
    path_for_all_objects_for_goal1 = (
        "/Users/ibtihajnaeem/Documents/version_control/thesis/detectAndTrajectoryPackage/assets"
        "/mmdetection3d/GOAL1/all_objects/")
    analyze_model = AnalyzeModel(goal=goal, path_for_all_objects=path_for_all_objects_for_goal1, )
    analyzer_utils = AnalyzerUtils()
    obj = AnalyzeResultsController(analyzeModel=analyze_model, analyzerUtils=analyzer_utils)
    all_keys_path = obj.get_keys().keys()
    print(all_keys_path)
    keys_list = list(all_keys_path)[0]
    print(keys_list)
    key_path = keys_list
    data = obj.compute(single_key_path=key_path)
    data.get_pkl_metrics()
    data.get_original_metrics()
    # data.print_all_metrics()
    print("-----differences-----")
    print(data.calculate_differences())


def compute_example_goal2():
    # SOLUTION FOR GOAL 2
    goal = GOAL.goal2
    path_for_all_objects_for_goal2 = (
        "/Users/ibtihajnaeem/Documents/version_control/thesis/detectAndTrajectoryPackage/assets"
        "/mmdetection3d/GOAL2/retry_all_objects/")
    analyze_model = AnalyzeModel(goal=goal, path_for_all_objects=path_for_all_objects_for_goal2, )
    analyzer_utils = AnalyzerUtils()
    obj = AnalyzeResultsController(analyzeModel=analyze_model, analyzerUtils=analyzer_utils)
    all_keys_path = obj.get_keys().keys()
    print(all_keys_path)
    keys_list = list(all_keys_path)[0]
    print(keys_list)
    key_path = keys_list
    data = obj.compute(single_key_path=key_path)
    data.get_pkl_metrics()
    data.get_original_metrics()
    # data.print_all_metrics()
    print("-----differences-----")
    print(data.calculate_differences())


compute_example_goal1()
compute_example_goal2()
