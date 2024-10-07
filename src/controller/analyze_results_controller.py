from model.analyze_model import AnalyzeModel
from enumerations import GOAL
import json, os
from glob import glob
from utils.class_implementations.analyzer_utils import AnalyzerUtils


class AnalyzeResultsController:
    def __init__(self, analyzeModel: AnalyzeModel, analyzerUtils: AnalyzerUtils):
        self.goal = analyzeModel.goal
        self.path_for_all_objects = analyzeModel.path_for_all_objects
        self.detector_results_path = analyzeModel.detector_results_path
        self.analyzerUtils = analyzerUtils

    def compute(self):
        goal = self.goal
        path_for_all_objects = self.path_for_all_objects
        detector_results_path = self.detector_results_path

        results_file_ap = ""
        results_file_pkl_crit = ""
        results_file_pkl = ""
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
        pkl_data = self.analyzerUtils.pkl_mean_ap_statistics(goal=goal, results_total=results_total,
                                                             detector_results_path=detector_results_path)
        original_data = self.analyzerUtils.mean_ap_statistics(goal=goal, results_total=results_total,
                                                              detector_results_path=detector_results_path)
        print(pkl_data)
        print(original_data)
