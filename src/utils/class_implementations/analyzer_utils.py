import json, os
from enumerations import GOAL
from model.Domain_models.detector_performance_metrics import DetectorPerformanceMetrics


def _get_data(goal, data):
    if goal == GOAL.goal1:
        return data['PKL_CRIT_RESULTS']
    else:
        return data['PKL_CRIT_RESULTS_GOAL2']


class AnalyzerUtils:
    @staticmethod
    def analyze_pkl_crit_goal(path, file_name, goal):
        f = open(path + "/" + file_name)
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        results = []
        new_data = _get_data(goal, data)
        for i in new_data:
            for d in i:
                results.append([d['min'],
                                d['max'],
                                d['mean'],
                                d['median'],
                                d['confidence'],
                                d['crit'],
                                d['D'],
                                d['R'],
                                d['T'],
                                ])

        return results

    @staticmethod
    def analyze_pkl(path, file_name):
        f = open(path + "/" + file_name)
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        results = []
        for i in data['PKL_RESULTS']:
            for d in i:
                results.append([d['min'],
                                d['max'],
                                d['mean'],
                                d['median'],
                                d['confidence'],
                                ])
        return results

    @staticmethod
    def analyze_ap(path, file_name):
        f = open(path + "/" + file_name)
        # returns JSON object as
        # a dictionary
        data = json.load(f)
        results = []
        for i in data['AP_RESULTS']:
            results.append([i['label_aps'],
                            i['label_aps_crit'],
                            i['DRT'],
                            i["mean_ap"],
                            i["mean_ap_crit"],
                            ])

        return results

    def process_directories(self, goal, directories, results_file_ap, results_file_pkl_crit, results_file_PKL):
        results_total = {}
        for d in directories:
            directory_name = d
            files = os.listdir(d)
            results_pkl_crit = []
            results_pkl = []
            results_ap = []
            for f in files:
                if f == results_file_pkl_crit:
                    results_pkl_crit = self.analyze_pkl_crit_goal(d, results_file_pkl_crit, goal)
                if f == results_file_PKL:
                    results_pkl = self.analyze_pkl(d, results_file_PKL)
                if f == results_file_ap:
                    results_ap = self.analyze_ap(d, results_file_ap)
            results_total.update({str(directory_name): [results_pkl, results_pkl_crit, results_ap]})
            return results_total

    @staticmethod
    def pkl_mean_ap_statistics(goal, results_total, path_for_single_key):
        d, r, t = 0, 0, 0
        mean_ap_max = 0.0
        if goal == GOAL.goal1:
            print("mean ap: {:.3f}".format(results_total[path_for_single_key][2][0][3]))
            for j in results_total[path_for_single_key][2]:
                if mean_ap_max <= j[4]:
                    mean_ap_max = j[4]
                    d, r, t = j[2]['D'], j[2]['R'], j[2]['T']
        mean_pkl = 10000
        saved_pkl = []
        print("-------")
        print(results_total)
        for j in results_total[path_for_single_key][1]:
            tmp = j[2]
            if tmp <= mean_pkl:
                mean_pkl = tmp
                saved_pkl = j
        print("min, max, lowest mean, median PKL crit, confidence, criticality, D, R, T")
        print(saved_pkl)

        lowest_mean = saved_pkl[2]
        saved_pkl_mean = saved_pkl

        # select target detector i, get lowest median PKL_CRIT

        median_pkl = 100000
        saved_pkl = []
        for j in results_total[path_for_single_key][1]:
            tmp = j[3]
            if tmp <= median_pkl:
                median_pkl = tmp
                saved_pkl = j
        print("min, max, mean, lowest median PKL crit, confidence, criticality, D, R, T")

        print(saved_pkl)

        lowest_median = saved_pkl[3]
        saved_pkl_median = saved_pkl

        max_pkl = 100000
        saved_pkl = []
        for j in results_total[path_for_single_key][1]:
            tmp = j[1]
            if tmp <= max_pkl:
                max_pkl = tmp
                saved_pkl = j
        print("min, lowest max, mean,median PKL crit, confidence, criticality, D, R, T")
        print(saved_pkl)

        lowest_max = saved_pkl[1]
        saved_pkl_max = saved_pkl

        print("\n\nlowest mean pkl: {:.3f} with {:.2f} and crit {:.2f}, ({}, {}, {})".format(lowest_mean,
                                                                                             saved_pkl_mean[4],
                                                                                             saved_pkl_mean[5],
                                                                                             saved_pkl_mean[6],
                                                                                             saved_pkl_mean[7],
                                                                                             saved_pkl_mean[8]))
        print("lowest median pkl: {:.3f} with {:.2f} and crit {:.2f}, ({}, {}, {})".format(lowest_median,
                                                                                           saved_pkl_median[4],
                                                                                           saved_pkl_median[5],
                                                                                           saved_pkl_median[6],
                                                                                           saved_pkl_median[7],
                                                                                           saved_pkl_median[8]))
        print("lowest max pkl: {:.3f} with {:.2f} and crit {:.2f}, ({}, {}, {})".format(lowest_max,
                                                                                        saved_pkl_max[4],
                                                                                        saved_pkl_max[5],
                                                                                        saved_pkl_max[6],
                                                                                        saved_pkl_max[7],
                                                                                        saved_pkl_max[8]))
        return DetectorPerformanceMetrics(
            goal=goal, mean_ap=mean_ap_max, mean_ap_max=mean_ap_max, mean_ap_max_criteria=(d, r, t),
            lowest_mean=lowest_mean, saved_pkl_mean=saved_pkl_mean, lowest_median=lowest_median,
            saved_pkl_median=saved_pkl_median, lowest_max=lowest_max, max_pkl=saved_pkl_max
        )

    @staticmethod
    def original_mean_ap_statistics(goal, results_total, detector_results_path):
        print("best results with original pkl, valid only for GOAL1\n")
        if goal == GOAL.goal1:
            mean_pkl = 10000
            saved_pkl = []
            for j in results_total[detector_results_path][0]:
                tmp = j[2]
                if tmp < mean_pkl:
                    mean_pkl = tmp
                    saved_pkl = j
            print("min, max,  lowest mean, median PKL, confidence")
            print(saved_pkl)
            lowest_mean = saved_pkl[2]
            saved_pkl_mean = saved_pkl

            # select target detector i, get lowest median PKL

            median_pkl = 10000
            saved_pkl = []
            for j in results_total[detector_results_path][0]:
                tmp = j[3]

                if tmp < median_pkl:
                    median_pkl = tmp
                    saved_pkl = j
            print("min, max, mean, lowest median PKL, confidence")
            print(saved_pkl)
            lowest_median = saved_pkl[3]
            saved_pkl_median = saved_pkl

            max_pkl = 10000
            saved_pkl = []
            for j in results_total[detector_results_path][0]:
                tmp = j[1]

                if tmp < max_pkl:
                    max_pkl = tmp
                    saved_pkl = j
            print("min, max, mean, lowest median PKL, confidence")
            print(saved_pkl)
            lowest_max = saved_pkl[1]
            saved_pkl_max = saved_pkl

            print("\n\nlowest mean pkl: {:.3f} with {:.2f} ".format(lowest_mean, saved_pkl_mean[4]))
            print("lowest median pkl: {:.3f} with {:.2f}".format(lowest_median, saved_pkl_median[4]))
            print("lowest max pkl: {:.3f} with {:.2f}".format(lowest_max, saved_pkl_max[4]))
            return DetectorPerformanceMetrics(
                goal=goal, mean_ap=0, mean_ap_max=0, mean_ap_max_criteria=0,
                lowest_mean=lowest_mean, saved_pkl_mean=saved_pkl_mean, lowest_median=lowest_median,
                saved_pkl_median=saved_pkl_median, lowest_max=lowest_max, max_pkl=saved_pkl_max
            )
