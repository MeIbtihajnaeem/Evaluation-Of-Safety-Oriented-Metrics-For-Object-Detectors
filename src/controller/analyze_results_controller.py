from model.DTOs.analyze_model import AnalyzeModel
from enumerations import GOAL
from glob import glob
from utils.class_implementations.analyzer_utils import AnalyzerUtils
from model.Domain_models.combined_detector_performance_metrics import CombinedDetectorPerformanceMetrics


class AnalyzeResultsController:
    def __init__(self, analyzeModel: AnalyzeModel, analyzerUtils: AnalyzerUtils):
        self.goal = analyzeModel.goal
        self.path_for_all_objects = analyzeModel.path_for_all_objects
        self.analyzerUtils = analyzerUtils
        self.result_total = {}

    def get_keys(self):
        goal = self.goal
        path_for_all_objects = self.path_for_all_objects
        if goal == GOAL.goal1:
            results_file_ap = 'ap_results.json'
            results_file_pkl_crit = 'pkl_crit_results_GOAL1.json'
            results_file_pkl = 'pkl_results.json'
        elif goal == GOAL.goal2:
            results_file_pkl_crit = "pkl_crit_results_GOAL2.json"
            results_file_ap = "None"
            results_file_pkl = "None"
        else:
            raise Exception("Goal Not found")
        directories = glob(path_for_all_objects + "/*/", recursive=True)
        results_total = self.analyzerUtils.process_directories(goal=goal, directories=directories,
                                                               results_file_ap=results_file_ap,
                                                               results_file_pkl_crit=results_file_pkl_crit,
                                                               results_file_PKL=results_file_pkl)
        self.result_total = results_total
        return results_total

    def compute(self, single_key_path: str):
        if not isinstance(single_key_path, str):
            raise ValueError(
                "Please provide a valid path")
        if self.result_total == {}:
            raise ValueError(
                "Please generate key first")
        goal = self.goal
        pkl_data = self.analyzerUtils.pkl_mean_ap_statistics(goal=goal, results_total=self.result_total,
                                                             path_for_single_key=single_key_path)

        if goal == GOAL.goal1:
            original_data = self.analyzerUtils.original_mean_ap_statistics(goal=goal, results_total=self.result_total,
                                                                           detector_results_path=single_key_path)
            return CombinedDetectorPerformanceMetrics(pkl_metrics=pkl_data, original_metrics=original_data)

        return CombinedDetectorPerformanceMetrics(pkl_metrics=pkl_data, original_metrics=None)
