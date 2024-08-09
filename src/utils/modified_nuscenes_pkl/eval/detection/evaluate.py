# nuScenes dev-kit.
# Code written by Holger Caesar & Oscar Beijbom, 2018.

import argparse
import json
import os
import random
import sys
import time
import math
import numpy as np
import torch
import itertools
from typing import Tuple, Dict, Any, List
from pyquaternion import Quaternion
from copy import deepcopy
# from pkl import calculate_pkl, test_pkl, pkl_print_visualizations, test_pkl_2
# from nuscenes.eval.common.utils import center_distance
# from ..common.data_classes import EvalBox
# from nuscenes.eval.common.data_classes import EvalBoxes
# from ..common.data_classes import EvalBoxes
# from ..common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
# from ..common.consolidated_imports import import_loaders
# from .algo import accumulate, calc_ap, calc_ap_crit, calc_tp
# from .render import summary_plot, summary_plot_crit, class_pr_curve, class_pr_curve_crit, \
#     class_tp_curve, dist_pr_curve, visualize_sample, visualize_sample_crit, visualize_sample_crit_r, \
#     visualize_sample_crit_d, visualize_sample_crit_t, visualize_sample_debug_1
# from ...utils.data_classes import Box
# from .data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList
# from ....pkl.planning_kl import calculate_pkl, test_pkl, pkl_print_visualizations, test_pkl_2


from utils.modified_nuscenes_pkl.eval.common.loaders import load_prediction, load_gt, filter_eval_boxes
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.constants import TP_METRICS
from utils.modified_nuscenes_pkl.pkl.consolidated_imports import import_for_eval as pkl_all_imports
from utils.modified_nuscenes_pkl.eval.common.consolidated_imports import import_loaders
# from ..common.consolidated_imports import import_loaders
from ..common.data_classes import EvalBoxes
from .consolidated_imports import import_algo, import_render
from .consolidated_imports import import_data_classes as detection_data_classes
from ...utils.consolidated_imports import import_all as nuscenes_utils
from utils.modified_nuscenes_pkl.eval.detection.data_classes import DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from utils.modified_nuscenes_pkl.eval.detection.algo import accumulate, calc_ap_crit
from utils.modified_nuscenes_pkl.eval.detection.render import summary_plot_crit, class_pr_curve_crit,visualize_sample_crit, visualize_sample_crit_r, visualize_sample_crit_d,visualize_sample_crit_t, visualize_sample_debug_1
calculate_pkl, test_pkl, pkl_print_visualizations, test_pkl_2 = pkl_all_imports()
DetectionConfig = detection_data_classes()
Box = nuscenes_utils()
add_center_dist = import_loaders()
calc_ap, calc_tp = import_algo()
summary_plot, class_pr_curve, class_tp_curve, visualize_sample, dist_pr_curve = import_render()

# from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_ap_crit, calc_tp
# from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
#     DetectionMetricDataList

# from .utils import json_to_csv
# from nuscenes.utils.geometry_utils import transform_matrix
# from nuscenes.eval.common.utils import boxes_to_sensor

#Done , Done
model_name = "None",
MAX_DISTANCE_OBJ = 0.0,
MAX_DISTANCE_INTERSECT = 0.0,
MAX_TIME_INTERSECT = 0.0
recall_type = "NONE"
gt_boxes = None
pred_boxes = None

gpuid = 0
bsz = 128
nworkers = 10


class DetectionEval:
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 model_name=None,
                 MAX_DISTANCE_OBJ: float = 100.0,
                 MAX_DISTANCE_INTERSECT: float = 101.0,
                 MAX_TIME_INTERSECT_OBJ: float = 102.0,
                 verbose: bool = True,
                 recall_type="PRED AL NUMERATORE",
                 nworkers=10,
                 bsz=128,
                 gpuid=0, output_dir=None, crit=-1000
                 ):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.verbose = verbose
        self.cfg = config
        self.model_name = model_name
        self.MAX_DISTANCE_OBJ = MAX_DISTANCE_OBJ
        self.MAX_DISTANCE_INTERSECT = MAX_DISTANCE_INTERSECT
        self.MAX_TIME_INTERSECT = MAX_TIME_INTERSECT_OBJ
        self.recall_type = recall_type
        DetectionBox.MAX_DISTANCE_OBJ = MAX_DISTANCE_OBJ
        DetectionBox.MAX_DISTANCE_INTERSECT = MAX_DISTANCE_INTERSECT
        DetectionBox.MAX_TIME_INTERSECT_OBJ = MAX_TIME_INTERSECT_OBJ
        self.nworkers = nworkers
        self.bsz = bsz
        self.gpuid = gpuid
        self.output_dir = output_dir
        self.crit = crit

        # Check result file exists.
        assert os.path.exists(result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if self.verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(nusc, self.result_path, self.cfg.max_boxes_per_sample,
                                                     DetectionBox, verbose=self.verbose)
        self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=self.verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if self.verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=self.verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=self.verbose)

    #        self.sample_tokens = self.gt_boxes.sample_tokens

    def evaluate(self) -> Tuple[DetectionMetrics, DetectionMetricDataList]:
        """
        Performs the actual evaluation.
        :return: A tuple of high-level and the raw metric data.
        """
        print("STARTING EVALUATION in evaluate(self)")
        start_time = time.time()

        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print('Accumulating metric data...')

        pred_boxes1, number = self.filter_boxes_criticality(self.pred_boxes, self.crit)
        print("predicted bboxes are {}".format(pred_boxes1))

        metric_data_list = DetectionMetricDataList()
        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(self.gt_boxes, pred_boxes1, class_name,
                                self.cfg.dist_fcn_callable, dist_th, path=self.output_dir,
                                model_name=self.model_name,
                                MAX_DISTANCE_OBJ=self.MAX_DISTANCE_OBJ,
                                MAX_DISTANCE_INTERSECT=self.MAX_DISTANCE_INTERSECT,
                                MAX_TIME_INTERSECT=self.MAX_TIME_INTERSECT,
                                recall_type=self.recall_type)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print('Calculating metrics...')
        metrics = DetectionMetrics(self.cfg)
        f = open(self.output_dir + "/AP_SUMMARY.txt", "a")
        f.write("Model;MAX_DISTANCE_OBJ;MAX_DISTANCE_INTERSECT;MAX_TIME_INTERSECT;class_name;dist_th;ap;ap_crit\n")
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
                ap_crit = calc_ap_crit(metric_data, self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap_crit(class_name, dist_th, ap_crit)
                f.write(str(self.model_name) +
                        ";" + str(self.MAX_DISTANCE_OBJ) +
                        ";" + str(self.MAX_DISTANCE_INTERSECT) +
                        ";" + str(self.MAX_TIME_INTERSECT) +
                        ";" + str(class_name) +
                        ";" + str(dist_th) +
                        ";" + str(ap) +
                        ";" + str(ap_crit) + "\n")

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)
        f.close()

        return metrics, metric_data_list

    def render(self, metrics: DetectionMetrics, md_list: DetectionMetricDataList) -> None:
        """
        Renders various PR and TP curves.
        :param metrics: DetectionMetrics instance.
        :param md_list: DetectionMetricDataList instance.
        """
        if self.verbose:
            print('Rendering PR and TP curves')

        def savepath(name):
            return os.path.join(self.plot_dir, name + '.pdf')

        def savepath_crit(name):
            return os.path.join(self.plot_dir, name + '_crit.pdf')

        summary_plot(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                     dist_th_tp=self.cfg.dist_th_tp, savepath=savepath('summary'))

        summary_plot_crit(md_list, metrics, min_precision=self.cfg.min_precision, min_recall=self.cfg.min_recall,
                          dist_th_tp=self.cfg.dist_th_tp, savepath=savepath_crit('summary'))

        for detection_name in self.cfg.class_names:
            class_pr_curve(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                           savepath=savepath(detection_name + '_pr'))

            class_pr_curve_crit(md_list, metrics, detection_name, self.cfg.min_precision, self.cfg.min_recall,
                                savepath=savepath(detection_name + '_crit_pr'))
            print(metrics)
            print(md_list)

            class_tp_curve(md_list, metrics, detection_name, self.cfg.min_recall, self.cfg.dist_th_tp,
                           savepath=savepath(detection_name + '_tp'))

        for dist_th in self.cfg.dist_ths:
            dist_pr_curve(md_list, metrics, dist_th, self.cfg.min_precision, self.cfg.min_recall,
                          savepath=savepath('dist_pr_' + str(dist_th)))

    def calc_sample_crit(self,
                         set_of_tokens: str,
                         conf_ths=[0.5],
                         dist_ths=[2.0],
                         crit_list=[1.0],
                         #                         boxes_gt_1,
                         #                         boxes_pred_1,
                         obj_classes_list=['car'],
                         verbose: bool = False,
                         model_loaded=False,  # an attempt to load the model outside the pkl
                         model_object=None,
                         ):
        """ Method for computing safety-oriented metrics data for predictions over a single sample
        :param sample_token: sample evaluated.
        :param save_path: directory to store results.
        :param verbose: print progress.
        """
        pkl_results = []
        pkl_crit_results = []

        if (obj_classes_list == []):
            print("Error: verify why obj_classes_list==[]")
            sys.exit(0)

        # Get boxes corresponding to sample
        #        boxes_gt = boxes_gt_1
        #        boxes_pred = boxes_pred_1
        #        if self.verbose:
        #            print("Object classes: {}".format(obj_classes_list))
        #            print("You set verbose=True; this check is taking long")
        #            counter_GT=0
        #            counter_PT=0
        #            for i in set_of_tokens:
        #                counter_GT=counter_GT+len(self.gt_boxes.serialize()[i])
        #                counter_PT=counter_PT+len(self.pred_boxes.serialize()[i])
        #            print("Number of GT boxes: {}".format(counter_GT))
        #            print("Number of PD boxes: {}".format(counter_PT))

        # Accumulate metric data for specific sample
        metric_data_list = DetectionMetricDataList()

        if verbose:
            print("computing accumulate for {} and {}".format(self.cfg.class_names, dist_ths))
        for class_name in self.cfg.class_names:
            for dist_th in dist_ths:
                if verbose:
                    print(" accumulate for {}, {} compute_crit_pkl".format(dist_th, class_name))
                md = accumulate(self.gt_boxes, self.pred_boxes, class_name,
                                self.cfg.dist_fcn_callable, dist_th,
                                MAX_DISTANCE_OBJ=self.MAX_DISTANCE_OBJ,
                                MAX_DISTANCE_INTERSECT=self.MAX_DISTANCE_INTERSECT,
                                MAX_TIME_INTERSECT=self.MAX_TIME_INTERSECT,
                                recall_type=self.recall_type,
                                verbose=False,
                                single_sample=False,
                                conf_th_sample=0.0),  # in use only if single_sample=True

                metric_data_list.set(class_name, dist_th, md)

        # Calculate metrics from the data.
        if verbose:
            print('Calculating metrics...')

        metrics = DetectionMetrics(self.cfg)

        # Compute APs.
        for class_name in self.cfg.class_names:
            for dist_th in dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                if (len(list(metric_data)) != 1):
                    print("error in evaluate.py, metric_data has length {}".format(len(list(metric_data))))
                    sys.exit(0)

                ap = calc_ap(metric_data[0], self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
                ap_crit = calc_ap_crit(metric_data[0], self.cfg.min_recall, self.cfg.min_precision)
                metrics.add_label_ap_crit(class_name, dist_th, ap_crit)

        # Compute TP metrics.
        for metric_name in TP_METRICS:
            metric_data = metric_data_list[(class_name, self.cfg.dist_th_tp)]
            if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                tp = np.nan
            elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                tp = np.nan
            else:
                tp = calc_tp(metric_data[0], self.cfg.min_recall, metric_name)
            metrics.add_label_tp(class_name, metric_name, tp)

        # metrics è di tipo DetectionMetrics
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()

        # Compute PKL
        device = torch.device(f'cuda:{self.gpuid}') if self.gpuid >= 0 \
            else torch.device('cpu')

        map_folder = '/home/notebook/nuscene/'
        nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                           map_name=map_name) for map_name in [
                         "singapore-hollandvillage",
                         "singapore-queenstown",
                         "boston-seaport",
                         "singapore-onenorth",
                     ]}

        # Filter BBs on confidence before PKL evaluation
        for conf in conf_ths:  # conf_ths: confidence scores 0.1, 0.2, .. 1.0  (usually from 0.4 to 0.6)
            # just to be sure self.pred_boxes is not changed
            box_tmp = deepcopy(self.pred_boxes)
            box_tmp, number_of_boxes = self.filter_boxes_confidence(pred_boxes=box_tmp, conf_th=conf)

            if verbose:
                print("pred_boxes is {}".format(self.pred_boxes))
                print("box_tmp is {}, that is pred_boxes with confidence threshold  > {}".format(box_tmp, conf))
                print("compute pkl for conf {}".format(conf))

            pkl = calculate_pkl(self.gt_boxes, box_tmp,
                                set_of_tokens, self.nusc,
                                nusc_maps, device,
                                nworkers=self.nworkers,
                                bsz=self.bsz,
                                plot_kextremes=0,
                                verbose=verbose,
                                model_loaded=model_loaded,  # an attempt to load the model outside the pkl
                                model_object=model_object,
                                )
            # store pkl results in a list
            pkl.update({"confidence": conf, "number_boxes": number_of_boxes})
            pkl_results.append(pkl)

            for crit_score in crit_list:
                # just to be sure boxes are not compromised
                box_tmp1 = deepcopy(box_tmp)
                # now pkl with reduced boxes, thanks to criticality crit_score
                box_tmp1, number_of_boxes = self.filter_boxes_criticality(pred_boxes=box_tmp1,
                                                                          crit=crit_score)
                if verbose:
                    print("box_tmp1 for criticality {} is : {} ".format(crit_score, box_tmp1))

                pkl = calculate_pkl(self.gt_boxes, box_tmp1,
                                    set_of_tokens, self.nusc,
                                    nusc_maps, device,
                                    nworkers=nworkers, bsz=16,
                                    plot_kextremes=0,
                                    verbose=verbose,
                                    model_loaded=model_loaded,  # an attempt to load the model outside the pkl
                                    model_object=model_object,
                                    )

                pkl.update({"confidence": conf, "crit": crit_score,
                            "D": self.MAX_DISTANCE_OBJ,
                            "R": self.MAX_DISTANCE_INTERSECT,
                            "T": self.MAX_TIME_INTERSECT,
                            "number_boxes": number_of_boxes})

                pkl_crit_results.append(pkl)
        return metrics_summary, pkl_results, pkl_crit_results

    # for GOAL 2 of the paper
    def calc_sample_crit_GOAL2(self,
                               set_of_tokens: str,
                               conf_ths=[0.5],
                               dist_ths=[2.0],
                               crit_list=[1.0],
                               #                         boxes_gt_1,
                               #                         boxes_pred_1,
                               obj_classes_list=['car'],
                               verbose: bool = False,
                               model_loaded=False,  # an attempt to load the model outside the pkl
                               model_object=None,
                               ):

        if (sorted(obj_classes_list) != sorted(self.cfg.class_names)):
            print("filter BB to objects {}".format(obj_classes_list))
            # Filter BBs on object class, for example "car"
            self.gt_boxes, self.pred_boxes = self.filter_boxes_class(gt_boxes=self.gt_boxes,
                                                                     pred_boxes=self.pred_boxes,
                                                                     classes=obj_classes_list)

        pkl_crit_results = []

        if (obj_classes_list == []):
            print("Error: verify why obj_classes_list==[]")
            sys.exit(0)

        # Compute PKL
        device = torch.device(f'cuda:{self.gpuid}') if self.gpuid >= 0 \
            else torch.device('cpu')

        map_folder = '/home/notebook/nuscene/'
        nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                           map_name=map_name) for map_name in [
                         "singapore-hollandvillage",
                         "singapore-queenstown",
                         "boston-seaport",
                         "singapore-onenorth",
                     ]}

        # box_tmp, number_of_boxes = self.filter_boxes_confidence(pred_boxes=box_tmp, conf_th=conf)

        # if verbose:
        #     print("pred_boxes is {}".format(self.pred_boxes))
        #     print("box_tmp is {}, that is pred_boxes with confidence threshold  > {}".format(box_tmp, conf))
        #     print("compute pkl for conf {}".format(conf))

        confidence_criticality = list(itertools.product(*[conf_ths, crit_list]))

        for confidence, criticality in confidence_criticality:
            # Filter BBs on confidence before PKL evaluation
            # for conf in conf_ths: #conf_ths: confidence scores 0.1, 0.2, .. 1.0
            # just to be sure self.pred_boxes is not changed
            box_tmp = deepcopy(self.pred_boxes)

            # now pkl with reduced boxes, thanks to criticality crit_score
            box_tmp, number_of_boxes = self.filter_boxes_confidence_criticalityGOAL2(pred_boxes=box_tmp,
                                                                                     conf_th=confidence,
                                                                                     crit=criticality)
            # if verbose:
            #     print("pred_boxes.boxes for criticality {} is  {}".format(crit_score, box_tmp))

            pkl = calculate_pkl(self.gt_boxes, box_tmp,
                                set_of_tokens, self.nusc,
                                nusc_maps, device,
                                nworkers=nworkers, bsz=16,
                                plot_kextremes=0,
                                verbose=verbose,
                                model_loaded=model_loaded,  # an attempt to load the model outside the pkl
                                model_object=model_object,
                                )

            pkl.update({"confidence": confidence,
                        "crit": criticality,
                        "D": self.MAX_DISTANCE_OBJ,
                        "R": self.MAX_DISTANCE_INTERSECT,
                        "T": self.MAX_TIME_INTERSECT,
                        "number_boxes": number_of_boxes})

            pkl_crit_results.append(pkl)
        return pkl_crit_results

    def calc_sample_crit_GOAL2_unique_param(self,
                                            listtoken: List[str],
                                            conf_th=0.4,  # confidence threshold
                                            crit=0.6,  # criticality
                                            obj_classes_list=[],  # filter boxes based on class
                                            verbose=False,
                                            image_counter=1,
                                            model_loaded=False,  # an attempt to load the model outside the pkl
                                            model_object=None,
                                            ):

        pkl = {}
        if (sorted(obj_classes_list) != sorted(self.cfg.class_names)):
            print("filter BB to objects {}".format(obj_classes_list))
            # Filter BBs on object class, for example "car"
            self.gt_boxes, self.pred_boxes = self.filter_boxes_class(gt_boxes=self.gt_boxes,
                                                                     pred_boxes=self.pred_boxes,
                                                                     classes=obj_classes_list)

        box_tmp = deepcopy(self.pred_boxes)
        # now pkl with reduced boxes, thanks to criticality crit_score
        box_tmp, number_of_boxes = self.filter_boxes_confidence_criticalityGOAL2(pred_boxes=box_tmp,
                                                                                 conf_th=conf_th,
                                                                                 crit=crit)

        if verbose:
            print("pred_boxes.boxes for criticality {} and threshold is  {}".format(crit, conf_th, box_tmp))

        # Compute PKL
        device = torch.device(f'cuda:{self.gpuid}') if self.gpuid >= 0 \
            else torch.device('cpu')

        map_folder = '/home/notebook/nuscene/'
        nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                           map_name=map_name) for map_name in [
                         "singapore-hollandvillage",
                         "singapore-queenstown",
                         "boston-seaport",
                         "singapore-onenorth",
                     ]}

        info, all_pkls, gtdist, preddist, gtxs, predxs, createdimages, gt_boxes, nusc_maps = pkl_print_visualizations(
            self.gt_boxes,
            box_tmp,
            listtoken, self.nusc,
            nusc_maps, device,
            nworkers=nworkers, bsz=16,
            plot_kextremes=image_counter,
            verbose=verbose,
            model_loaded=model_loaded,
            model_object=model_object)

        pkl.update({"confidence": conf_th,
                    "crit": crit,
                    "D": self.MAX_DISTANCE_OBJ,
                    "R": self.MAX_DISTANCE_INTERSECT,
                    "T": self.MAX_TIME_INTERSECT,
                    "number_boxes": number_of_boxes})

        return pkl, info, all_pkls, gtdist, preddist, gtxs, predxs, createdimages

    def calc_sample_crit_GOAL3(self,
                               listtoken: List[str],
                               conf_th=0.4,  # confidence threshold
                               crit=0.6,  # criticality
                               correction_factor=None,
                               obj_classes_list=[],  # filter boxes based on class
                               verbose=False,
                               model_loaded=False,  # an attempt to load the model outside the pkl
                               model_object=None,
                               ):

        if (sorted(obj_classes_list) != sorted(self.cfg.class_names)):
            print("filter BB to objects {}".format(obj_classes_list))
            # Filter BBs on object class, for example "car"
            self.gt_boxes, self.pred_boxes = self.filter_boxes_class(gt_boxes=self.gt_boxes,
                                                                     pred_boxes=self.pred_boxes,
                                                                     classes=obj_classes_list)

        box_tmp = deepcopy(self.pred_boxes)
        # now pkl with reduced boxes, thanks to criticality crit_score
        box_tmp, number_of_boxes = self.filter_boxes_confidence_criticalityGOAL3(pred_boxes=box_tmp,
                                                                                 filtering_condition=correction_factor,
                                                                                 conf_th=conf_th,
                                                                                 crit=crit)
        if verbose:
            print("pred_boxes.boxes for criticality {} and threshold is  {}".format(crit, conf_th, box_tmp))

        # Compute PKL
        device = torch.device(f'cuda:{self.gpuid}') if self.gpuid >= 0 \
            else torch.device('cpu')

        map_folder = '/home/notebook/nuscene/'
        nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                           map_name=map_name) for map_name in [
                         "singapore-hollandvillage",
                         "singapore-queenstown",
                         "boston-seaport",
                         "singapore-onenorth",
                     ]}

        pkl = calculate_pkl(self.gt_boxes, box_tmp,
                            listtoken, self.nusc,
                            nusc_maps, device,
                            nworkers=nworkers, bsz=16,
                            plot_kextremes=0,
                            verbose=verbose,
                            model_loaded=model_loaded,  # an attempt to load the model outside the pkl
                            model_object=model_object,
                            )

        pkl.update({"confidence": conf_th,
                    "crit": crit,
                    "correction factor": correction_factor,
                    "D": self.MAX_DISTANCE_OBJ,
                    "R": self.MAX_DISTANCE_INTERSECT,
                    "T": self.MAX_TIME_INTERSECT,
                    "number_boxes": number_of_boxes})

        return pkl

    def pkl_test_2(self, listtoken: List[str],
                   conf_th=0.4,  # confidence threshold
                   crit=0.6,  # criticality
                   correction_factor=1,
                   obj_classes_list=[],  # filter boxes based on class
                   verbose=False,
                   model_loaded=False,  # an attempt to load the model outside the pkl
                   model_object=None, plot_kextremes=1
                   ):

        box_tmp = deepcopy(self.pred_boxes)
        # now pkl with reduced boxes, thanks to criticality crit_score
        box_tmp, number_of_boxes = self.filter_boxes_confidence_criticalityGOAL3(pred_boxes=box_tmp,
                                                                                 filtering_condition=correction_factor,
                                                                                 conf_th=conf_th, crit=crit)
        if verbose:
            print("pred_boxes.boxes for criticality {} and threshold is  {}".format(crit, conf_th, box_tmp))

        # Compute PKL
        device = torch.device(f'cuda:{self.gpuid}') if self.gpuid >= 0 \
            else torch.device('cpu')

        map_folder = '/home/notebook/nuscene/'
        nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                           map_name=map_name) for map_name in [
                         "singapore-hollandvillage",
                         "singapore-queenstown",
                         "boston-seaport",
                         "singapore-onenorth",
                     ]}

        info, all_pkls, gtdist, preddist, gtxs, predxs, worst_ixes, gt_boxes, nusc, nusc_maps, dataset = test_pkl_2(
            self.gt_boxes, box_tmp,
            listtoken, self.nusc,
            nusc_maps, device,
            nworkers=nworkers, bsz=16,
            plot_kextremes=plot_kextremes,
            verbose=verbose,
            model_loaded=model_loaded,  # an attempt to load the model outside the pkl
            model_object=model_object,
        )

        info.update({"confidence": conf_th,
                     "crit": crit,
                     "correction factor": correction_factor,
                     "D": self.MAX_DISTANCE_OBJ,
                     "R": self.MAX_DISTANCE_INTERSECT,
                     "T": self.MAX_TIME_INTERSECT,
                     "number_boxes": number_of_boxes})

        return info, all_pkls, gtdist, preddist, gtxs, predxs, worst_ixes, gt_boxes, nusc, nusc_maps, dataset

        # this is for GOAL 3, when I need to print images

    def calc_sample_crit_GOAL3_visualization(self,
                                             listtoken: List[str],
                                             conf_th=0.4,  # confidence threshold
                                             crit=0.6,  # criticality
                                             correction_factor=None,
                                             obj_classes_list=[],  # filter boxes based on class
                                             verbose=False,
                                             image_counter=1,
                                             model_loaded=False,  # an attempt to load the model outside the pkl
                                             model_object=None,
                                             ):

        pkl = {}
        if (sorted(obj_classes_list) != sorted(self.cfg.class_names)):
            print("filter BB to objects {}".format(obj_classes_list))
            # Filter BBs on object class, for example "car"
            self.gt_boxes, self.pred_boxes = self.filter_boxes_class(gt_boxes=self.gt_boxes,
                                                                     pred_boxes=self.pred_boxes,
                                                                     classes=obj_classes_list)

        box_tmp = deepcopy(self.pred_boxes)
        # now pkl with reduced boxes, thanks to criticality crit_score
        box_tmp, number_of_boxes = self.filter_boxes_confidence_criticalityGOAL3(pred_boxes=box_tmp,
                                                                                 filtering_condition=correction_factor,
                                                                                 conf_th=conf_th, crit=crit)
        if verbose:
            print("pred_boxes.boxes for criticality {} and threshold is  {}".format(crit, conf_th, box_tmp))

        # Compute PKL
        device = torch.device(f'cuda:{self.gpuid}') if self.gpuid >= 0 \
            else torch.device('cpu')

        map_folder = '/home/notebook/nuscene/'
        nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                           map_name=map_name) for map_name in [
                         "singapore-hollandvillage",
                         "singapore-queenstown",
                         "boston-seaport",
                         "singapore-onenorth",
                     ]}

        info, all_pkls, gtdist, preddist, gtxs, predxs, createdimages, gt_boxes, nusc_maps = pkl_print_visualizations(
            self.gt_boxes,
            box_tmp,
            listtoken, self.nusc,
            nusc_maps, device,
            nworkers=nworkers, bsz=16,
            plot_kextremes=image_counter,
            verbose=verbose,
            model_loaded=model_loaded,
            model_object=model_object)

        pkl.update({"confidence": conf_th,
                    "crit": crit,
                    "correction factor": correction_factor,
                    "D": self.MAX_DISTANCE_OBJ,
                    "R": self.MAX_DISTANCE_INTERSECT,
                    "T": self.MAX_TIME_INTERSECT,
                    "number_boxes": number_of_boxes})

        return pkl, info, all_pkls, gtdist, preddist, gtxs, predxs, createdimages, gt_boxes, nusc_maps

    # to implement GOAL 1 of the paper
    def safety_metric_evaluation(self,
                                 list_of_tokens: List[str],
                                 conf_th_list: float = [0.4],
                                 dist_list=[],
                                 obj_classes_list=[],
                                 crit_list=[],
                                 render_images=False,
                                 verbose=False,
                                 model_loaded=True,  # an attempt to load the model outside the pkl
                                 model_object=None,
                                 ):
        """ High-level function comprising functionality related to evaluating over single samples
        """

        #        boxes_gt = EvalBoxes()
        #        boxes_pred = EvalBoxes()
        #        for sample_token in list_of_tokens:
        #            boxes_gt.add_boxes(sample_token, self.gt_boxes.boxes[sample_token])
        #            boxes_pred.add_boxes(sample_token, self.pred_boxes.boxes[sample_token])

        # Filter BBs on confidence before PKL evaluation
        #            boxes_pred = self.filter_boxes_confidence(pred_boxes=boxes_pred, conf_th=conf_th_sample, obj_classes_list)

        if (sorted(obj_classes_list) != sorted(self.cfg.class_names)):
            print("filter BB to objects {}".format(obj_classes_list))
            # Filter BBs on object class, for example "car"
            self.gt_boxes, self.pred_boxes = self.filter_boxes_class(gt_boxes=self.gt_boxes,
                                                                     pred_boxes=self.pred_boxes,
                                                                     classes=obj_classes_list)

        ## Compute criticalities and PKT, return results
        return self.calc_sample_crit(set_of_tokens=list_of_tokens,
                                     conf_ths=conf_th_list,
                                     dist_ths=dist_list,
                                     crit_list=crit_list,
                                     #                                     boxes_gt_1=self.gt_boxes,
                                     #                                     boxes_pred_1=self.pred_boxes,
                                     obj_classes_list=obj_classes_list,
                                     verbose=verbose,
                                     model_loaded=model_loaded,  # an attempt to load the model outside the pkl
                                     model_object=model_object,
                                     )

    # to implement GOAL 2 of the paper
    def safety_metric_evaluation_GOAL2(self,
                                       list_of_tokens: List[str],
                                       conf_th_list: float = [0.4],
                                       dist_list=[],
                                       obj_classes_list=[],
                                       crit_list=[],
                                       render_images=False,
                                       verbose=False,
                                       model_loaded=False,  # an attempt to load the model outside the pkl
                                       model_object=None,
                                       ):
        """ High-level function comprising functionality related to evaluating over single samples
        """
        if (sorted(obj_classes_list) != sorted(self.cfg.class_names)):
            print("filter BB to objects {}".format(obj_classes_list))
            # Filter BBs on object class, for example "car"
            self.gt_boxes, self.pred_boxes = self.filter_boxes_class(gt_boxes=self.gt_boxes,
                                                                     pred_boxes=self.pred_boxes,
                                                                     classes=obj_classes_list)

        ## Compute criticalities and PKT, return results
        return self.calc_sample_crit_GOAL2(set_of_tokens=list_of_tokens,
                                           conf_ths=conf_th_list,
                                           dist_ths=dist_list,
                                           crit_list=crit_list,
                                           #                                     boxes_gt_1=self.gt_boxes,
                                           #                                     boxes_pred_1=self.pred_boxes,
                                           obj_classes_list=obj_classes_list,
                                           verbose=verbose,
                                           model_loaded=model_loaded,  # an attempt to load the model outside the pkl
                                           model_object=model_object,
                                           )

    def filter_boxes_confidence(self, pred_boxes, conf_th: float = 0.15):
        """
        Filter Pred boxes on a confidence threshold.
        :param pred_boxes: boxes to filter.
        :param conf_th: confidence threshold.
        """
        counter = 0
        for ind, sample_token in enumerate(pred_boxes.sample_tokens):
            pred_boxes.boxes[sample_token] = [box for box in pred_boxes[sample_token] if
                                              box.detection_score >= conf_th]
            counter = counter + len(pred_boxes.boxes[sample_token])

        return pred_boxes, counter

    def filter_boxes_criticality(self, pred_boxes, crit: float = 0.15):
        """
        Filter Pred boxes on criticality threshold.
        :param pred_boxes: boxes to filter.
        :param conf_th: confidence threshold.
        """
        counter = 0
        for ind, sample_token in enumerate(pred_boxes.sample_tokens):
            pred_boxes.boxes[sample_token] = [box for box in pred_boxes[sample_token] if
                                              box.crit >= crit]
            counter = counter + len(pred_boxes.boxes[sample_token])

        return pred_boxes, counter

    def filter_boxes_confidence_criticalityGOAL2(self, pred_boxes, conf_th: float = 0.4, crit: float = 0.8):
        """
        Filter Pred boxes according to GOAL 2: confidence threshold is applied only if criticality is lower than a given threshold.
        :param pred_boxes: boxes to filter.
        :param conf_th: confidence threshold.
        """
        counter = 0
        for ind, sample_token in enumerate(pred_boxes.sample_tokens):
            pred_boxes.boxes[sample_token] = [box for box in pred_boxes[sample_token] if
                                              (box.crit > crit or box.detection_score >= conf_th)]
            counter = counter + len(pred_boxes.boxes[sample_token])

        return pred_boxes, counter

    def filter_boxes_confidence_criticalityGOAL3(self, pred_boxes, filtering_condition=None, conf_th: float = 0.4,
                                                 crit: float = 0.8):
        """
        Filter Pred boxes according to GOAL 3:
        when to keep boxes:
         EXPLAINED IN filtering_condition function, passed as input
        :param filtering_condition : function to filter boxes
        :param pred_boxes: boxes to filter.
        :param conf_th: confidence threshold.
        """
        counter = 0
        if (filtering_condition == None):
            for ind, sample_token in enumerate(pred_boxes.sample_tokens):
                pred_boxes.boxes[sample_token] = [box for box in pred_boxes[sample_token] if
                                                  box.detection_score >= conf_th]
                counter = counter + len(pred_boxes.boxes[sample_token])
            return pred_boxes, counter

        for ind, sample_token in enumerate(pred_boxes.sample_tokens):
            pred_boxes.boxes[sample_token] = [box for box in pred_boxes[sample_token] if
                                              filtering_condition(box.crit, box.detection_score, crit, conf_th)]
            counter = counter + len(pred_boxes.boxes[sample_token])

        return pred_boxes, counter

    def filter_boxes_class(self, gt_boxes, pred_boxes, classes: List[str] = ["car"]):
        """
        Filter GT and Pred boxes to only include a set of object classes.
        :param pred_boxes: boxes to filter.
        :param classes: list of object class names.
        """

        for ind, sample_token in enumerate(pred_boxes.sample_tokens):
            pred_boxes.boxes[sample_token] = [box for box in pred_boxes[sample_token] if
                                              box.detection_name in classes]

        for ind, sample_token in enumerate(gt_boxes.sample_tokens):
            gt_boxes.boxes[sample_token] = [box for box in gt_boxes[sample_token] if
                                            box.detection_name in classes]

        return gt_boxes, pred_boxes

    def test_class(self, list_of_tokens: List[str],
                   conf_th_list: float = [0.4],
                   dist_list=[0.5],
                   obj_classes_list=[],
                   crit_list=[0.6],
                   render_images=False,
                   verbose=False,
                   visualize_only=False,
                   plot_kextremes=0
                   ) -> None:

        if (sorted(obj_classes_list) != sorted(self.cfg.class_names)):
            print("filter BB to objects {}".format(obj_classes_list))
            # Filter BBs on object class, for example "car"
            self.gt_boxes, self.pred_boxes = self.filter_boxes_class(gt_boxes=self.gt_boxes,
                                                                     pred_boxes=self.pred_boxes,
                                                                     classes=obj_classes_list)

        set_of_tokens = list_of_tokens
        conf_ths = conf_th_list
        dist_ths = dist_list
        crit_list = crit_list
        obj_classes_list = obj_classes_list
        pkl_crit_results = []
        # Compute PKL
        device = torch.device(f'cuda:{self.gpuid}') if self.gpuid >= 0 \
            else torch.device('cpu')

        map_folder = '/home/notebook/nuscene/'
        nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                                           map_name=map_name) for map_name in [
                         "singapore-hollandvillage",
                         "singapore-queenstown",
                         "boston-seaport",
                         "singapore-onenorth",
                     ]}

        if verbose:
            print("pred_boxes is {}".format(self.pred_boxes))
            print("compute pkl for conf {}".format(conf_th_list))

        box_tmp = deepcopy(self.pred_boxes)

        # now pkl with reduced boxes, thanks to criticality crit_score
        box_tmp, number_of_boxes = self.filter_boxes_confidence(pred_boxes=box_tmp, conf_th=conf_th_list[0])
        #       print("box_tmp is {}, that is pred_boxes with confidence threshold  > {}".format(box_tmp, conf))
        if verbose:
            print("pred_boxes.boxes for criticality {} is {} ".format(crit_list[0], box_tmp))

        info, all_pkls, gtdist, preddist, gtxs, predxs, worst_ixes = test_pkl(self.gt_boxes, box_tmp,
                                                                              set_of_tokens, self.nusc,
                                                                              nusc_maps, device,
                                                                              nworkers=nworkers, bsz=16,
                                                                              plot_kextremes=plot_kextremes,
                                                                              verbose=verbose,
                                                                              visualize_only=False)

        return info, all_pkls, gtdist, preddist, gtxs, predxs, worst_ixes

    def add_FP(self, sample_token: str, pos: Tuple[float, float], size: Tuple[float, float, float],
               match_ego_speed=False):
        """
        Add a FP prediction in coordinates in coordinates coords wrt. ego reference frame
        :param sample_token: Sample to operate on.
        :param pos: Tuple[x, y]. 2D coordinates of FP in relation to ego, where
        positive x and y point to right and front of ego, respectively. z coordinate is set equal to ego z.
        :param size: Tuple[h,l,w]. Size of injected BB.
        :param match_ego_speed: Boolean. Whether to match velocity of ego or to have null-velocity.
        """
        print("Adding FP at position {} relative to ego at sample {}".format(pos, sample_token))

        # Get ego reference
        sample = self.nusc.get('sample', sample_token)
        sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_translation = pose_record['translation']
        ego_rotation = pose_record['rotation']
        ego_speed = self.pred_boxes[sample_token][0].ego_speed
        # Create FP box
        fp_box = DetectionBox(
            sample_token=sample_token,
            translation=ego_translation,
            rotation=ego_rotation,
            size=size,
            detection_score=0.99,
            num_pts=25,
            attribute_name='vehicle.moving' if match_ego_speed == True else 'vehicle.stopped',
            velocity=(ego_speed[0], ego_speed[1]) if match_ego_speed == True else (0, 0),
            nusc=self.nusc,
        )

        # Create Box instance.
        box = Box(center=fp_box.translation, size=fp_box.size, orientation=Quaternion(fp_box.rotation),
                  velocity=(fp_box.velocity[0], fp_box.velocity[1], 0),
                  name=fp_box.detection_name, crit=fp_box.crit, crit_t=fp_box.crit_t, crit_r=fp_box.crit_r,
                  crit_d=fp_box.crit_d)
        # Move box to ego vehicle coord system.
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        # Place FP at pos coordinates (note: x and y coordinates are in opposite positions in ego frame: trans=(y,x,z))
        box.center[0] += pos[1]
        box.center[1] -= pos[0]  # positive x-axis to right

        # Transform to global ref frame
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))

        # Re-initialize Detection-Box to match new translation (to get correct criticalities)
        fp_box_m = DetectionBox(
            sample_token=sample_token,
            translation=box.center,
            rotation=ego_rotation,
            size=size,
            detection_score=0.99,
            num_pts=25,
            attribute_name='vehicle.moving' if match_ego_speed == True else 'vehicle.stopped',
            velocity=(ego_speed[0], ego_speed[1]) if match_ego_speed == True else (0, 0),
            nusc=self.nusc,
        )

        # add box
        self.pred_boxes.add_boxes(sample_token, [fp_box_m])

    def add_FN(self, sample_token: str, dist: float = 10, remove_all: bool = True, classes: List[str] = ['car']):
        """
        Add FN to predictions by removing BB that is less than dist from ego
        :param sample_token: sample token.
        :param dist: distance within which predictions are removed.
        :param remove_all: remove all BBs that fit criteria dist(ego, pred)<10 (or first one detected).
        :param object classes to consider when removing BBs, e.g. ['car', 'truck']
        """

        print("Removing {0} within {1}m from ego at sample {2}".format("all BBs" if remove_all else "first found BB",
                                                                       dist, sample_token))

        def bbs_dist(trans_1, trans_2):
            """ Return Euclidean dist between BB centers """
            return math.sqrt(
                (trans_2[0] - trans_1[0]) ** 2 + (trans_2[1] - trans_1[1]) ** 2 + (trans_2[2] - trans_1[2]) ** 2)

        sample = self.nusc.get('sample', sample_token)
        sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        ego_translation = pose_record['translation']
        it = 0
        shortest_dist = 999999.9
        shortest_dist_idx = None

        while it < len(self.pred_boxes[sample_token]):
            # Remove correctly predicted boxes within dist
            distance_pred = bbs_dist(ego_translation, self.pred_boxes[sample_token][it].translation)
            if distance_pred < dist and self.pred_boxes[sample_token][it].detection_name in classes:
                # Create Box instance for pred_box and transform to ego frame.
                pred_box = self.pred_boxes[sample_token][it]
                min_dist = 999999.9

                # Check if box is match (if not, cant be FN)
                for gt_idx, gt_box in enumerate(self.gt_boxes[pred_box.sample_token]):
                    # Find closest match among ground truth boxes
                    if gt_box.detection_name in classes:
                        this_distance = self.cfg.dist_fcn_callable(gt_box, pred_box)
                        if this_distance < min_dist:
                            min_dist = this_distance

                # If the closest match is close enough according to threshold we have a match!
                is_match = min_dist < 2.0  # dist_th 2.0
                if not is_match:
                    it += 1
                    continue

                # Transform to ego ref frame and check location
                box = Box(center=pred_box.translation, size=pred_box.size, orientation=Quaternion(pred_box.rotation),
                          velocity=(pred_box.velocity[0], pred_box.velocity[1], 0))
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                if remove_all:
                    del self.pred_boxes[sample_token][it]
                    continue

                if distance_pred < shortest_dist:
                    shortest_dist = distance_pred
                    shortest_dist_idx = it

            it += 1
        if not remove_all and shortest_dist_idx:
            del self.pred_boxes[sample_token][shortest_dist_idx]

    def main(self,
             plot_examples: int = 0,
             render_curves: bool = True,
             model_name="None",
             MAX_DISTANCE_OBJ=0.0,
             MAX_DISTANCE_INTERSECT=0.0,
             MAX_TIME_INTERSECT=0.0,
             recall_type="NONE") -> Dict[str, Any]:

        self.model_name = model_name
        self.MAX_DISTANCE_OBJ = MAX_DISTANCE_OBJ
        self.MAX_DISTANCE_INTERSECT = MAX_DISTANCE_INTERSECT
        self.MAX_TIME_INTERSECT = MAX_TIME_INTERSECT
        self.recall_type = recall_type
        """
        Main function that loads the evaluation code, visualizes samples, runs the evaluation and renders stat plots.
        :param plot_examples: How many example visualizations to write to disk.
        :param render_curves: Whether to render PR and TP curves to disk.
        :return: A dict that stores the high-level metrics and meta data.
        """
        print("STARTING EVALUATION in main (self)")

        if plot_examples > 0:
            # Select a random but fixed subset to plot.
            random.seed(42)
            sample_tokens = list(self.sample_tokens)
            random.shuffle(sample_tokens)
            sample_tokens = sample_tokens[:plot_examples]

            # Create images for debug_1 (ground truth only) images
            example_dir = os.path.join(self.output_dir, 'examples_gt_only')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample_debug_1(self.nusc,
                                         sample_token,
                                         self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                         # Don't render test GT.
                                         self.pred_boxes,
                                         eval_range=max(self.cfg.class_range.values()),
                                         savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

            # Visualize samples without crit
            example_dir = os.path.join(self.output_dir, 'examples_clean')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample(self.nusc,
                                 sample_token,
                                 self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                 # Don't render test GT.
                                 self.pred_boxes,
                                 eval_range=max(self.cfg.class_range.values()),
                                 savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

            # Visualize samples with crit
            example_dir = os.path.join(self.output_dir, 'examples_crit')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample_crit(self.nusc,
                                      sample_token,
                                      self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                      # Don't render test GT.
                                      self.pred_boxes,
                                      eval_range=max(self.cfg.class_range.values()),
                                      savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

            # Visualize samples with crit_r
            example_dir = os.path.join(self.output_dir, 'examples_crit_r')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample_crit_r(self.nusc,
                                        sample_token,
                                        self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                        # Don't render test GT.
                                        self.pred_boxes,
                                        eval_range=max(self.cfg.class_range.values()),
                                        savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

            # Visualize samples with crit_d
            example_dir = os.path.join(self.output_dir, 'examples_crit_d')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample_crit_d(self.nusc,
                                        sample_token,
                                        self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                        # Don't render test GT.
                                        self.pred_boxes,
                                        eval_range=max(self.cfg.class_range.values()),
                                        savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

            # Visualize samples with crit_t
            example_dir = os.path.join(self.output_dir, 'examples_crit_t')
            if not os.path.isdir(example_dir):
                os.mkdir(example_dir)
            for sample_token in sample_tokens:
                visualize_sample_crit_t(self.nusc,
                                        sample_token,
                                        self.gt_boxes if self.eval_set != 'test' else EvalBoxes(),
                                        # Don't render test GT.
                                        self.pred_boxes,
                                        eval_range=max(self.cfg.class_range.values()),
                                        savepath=os.path.join(example_dir, '{}.png'.format(sample_token)))

        # Run evaluation.
        metrics, metric_data_list = self.evaluate()

        # Render PR and TP curves.
        if render_curves:
            self.render(metrics, metric_data_list)

        # Dump the metric data, meta and metrics to disk.
        if self.verbose:
            print('Saving metrics to: %s' % self.output_dir)
        metrics_summary = metrics.serialize()
        metrics_summary['meta'] = self.meta.copy()
        with open(os.path.join(self.output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        with open(os.path.join(self.output_dir, 'metrics_details.json'), 'w') as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        #        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        #       for tp_name, tp_val in metrics_summary['tp_errors'].items():
        #           print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        #        print('NDS: %.4f' % (metrics_summary['nd_score']))
        #        print('Eval time: %.1fs' % metrics_summary['eval_time'])

        # Print per-class metrics.
        #        print()
        #        print('Per-class results:')
        #        print('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']

        #        for class_name in class_aps.keys():
        #            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
        #                  % (class_name, class_aps[class_name],
        #                     class_tps[class_name]['trans_err'],
        #                     class_tps[class_name]['scale_err'],
        #                     class_tps[class_name]['orient_err'],
        #                     class_tps[class_name]['vel_err'],
        #                     class_tps[class_name]['attr_err']))

        return metrics_summary


# class NuScenesEval(DetectionEval):
#     """
#     Dummy class for backward-compatibility. Same as DetectionEval.
#     """


if __name__ == "__main__":

    print("STARTING EVALUATION in NuScene Eval -- should not be used")

    # Settings.
    parser = argparse.ArgumentParser(description='Evaluate nuScenes detection results.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('result_path', type=str, help='The submission as a JSON file.')
    parser.add_argument('--output_dir', type=str, default='~/nuscenes-metrics',
                        help='Folder to store result metrics, graphs and example visualizations.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes',
                        help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--config_path', type=str, default='',
                        help='Path to the configuration file.'
                             'If no path given, the CVPR 2019 configuration will be used.')
    parser.add_argument('--plot_examples', type=int, default=10,
                        help='How many example visualizations to write to disk.')
    parser.add_argument('--render_curves', type=int, default=1,
                        help='Whether to render PR and TP curves to disk.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Whether to print to stdout.')
    args = parser.parse_args()

    result_path_ = os.path.expanduser(args.result_path)
    output_dir_ = os.path.expanduser(args.output_dir)
    eval_set_ = args.eval_set
    dataroot_ = args.dataroot
    version_ = args.version
    config_path = args.config_path
    plot_examples_ = args.plot_examples
    render_curves_ = bool(args.render_curves)
    verbose_ = bool(args.verbose)

    if config_path == '':
        cfg_ = config_factory('detection_cvpr_2019')
    else:
        with open(config_path, 'r') as _f:
            cfg_ = DetectionConfig.deserialize(json.load(_f))

    nusc_ = NuScenes(version=version_, verbose=verbose_, dataroot=dataroot_)
    nusc_eval = DetectionEval(nusc_, config=cfg_, result_path=result_path_, eval_set=eval_set_,
                              output_dir=output_dir_, verbose=verbose_)
    nusc_eval.main(plot_examples=plot_examples_, render_curves=render_curves_)
