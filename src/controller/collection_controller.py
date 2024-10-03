from enumerations import GOAL
from model.settings_model import SettingsModel

import json
from utils.class_implementations.collection_utils import CollectionUtils

import os
from utils.file_utils import FileUtils


class CollectionController:

    def __init__(self, settingsModel: SettingsModel, file_util: FileUtils, ColUtils: CollectionUtils):
        self.settings_model = settingsModel
        self.ColUtils = ColUtils
        self.__eval = settingsModel.scene_for_eval_set
        self.__val = settingsModel.scene_for_eval_set
        self.__scenes_list = self.ColUtils.get_scenes_list(settingsModel.nuscenes,
                                                           self.settings_model.scene_for_eval_set)
        self.__validation_samples = self.ColUtils.get_validation_samples(self.__scenes_list, settingsModel.nuscenes)
        self.__list_token = self.ColUtils.get_list_token(validation_samples=self.__validation_samples)
        self.file_util = file_util
        if not os.path.exists(settingsModel.results_path):
            os.makedirs(settingsModel.results_path)
        with open(settingsModel.results_path + "token_list.json", "w") as outfile:
            json.dump(self.__validation_samples, outfile)

    def run(self):
        settings_model = self.settings_model
        if settings_model.goal.__eq__(GOAL.goal1):
            self._compute_goal1()
        else:
            self._compute_goal2()
        confidence_threshold, criticality_threshold, d, r, t = self._compute_selected_precision_recall()
        data = self.file_util.read_data_from_file(
            settings_model.save_build_precision_recall_curve_data + "metrics_details.json")
        normal_05, normal_1, normal_2, normal_4 = self.ColUtils.plot_normal(data, confidence_threshold)
        save_in = settings_model.notebook_home + 'ttpkl/results/' + settings_model.goal.name + '/all_objects/' + settings_model.detector.value + '/PrecisionRecall/ReducedBBoxes/'  # will save all the data, that we need to explore to build precision recall curves
        self._compute_selected_precision_recall(critic=criticality_threshold, save_in=save_in)
        data = self.file_util.read_data_from_file(
            settings_model.save_build_precision_recall_curve_data + "metrics_details.json")
        special_05, special_1, special_2, special_4 = self.ColUtils.plot_special(data, confidence_threshold, save_in, d,
                                                                                 r, t)
        self.ColUtils.plot_normal_and_special(special_05, special_1, special_2, special_4, normal_05, normal_1,
                                              normal_2,
                                              normal_4, save_in, confidence_threshold,
                                              self.settings_model.detector.value, d, r, t)

    def _compute_selected_precision_recall(self, critic=None, save_in=None):
        print("Computation Started")
        settings_model = self.settings_model
        detector = self.settings_model.detector
        detector_obj = detector.get_configuration()
        if critic is None:
            critic = -10
        if save_in is None:
            save_in = settings_model.save_build_precision_recall_curve_data
        d = detector_obj.d
        r = detector_obj.r
        t = detector_obj.t
        confidence_threshold = detector_obj.confidence_threshold
        criticality_threshold = detector_obj.criticality_threshold
        dt = self.ColUtils.create_dt(nuscenes=settings_model.nuscenes, conf_value=settings_model.conf_value,
                                     detector_file=settings_model.detector_file, val='val', model=detector.value, d=d,
                                     r=r,
                                     t=t, verbose=True, n_workers=settings_model.n_workers, bsz=settings_model.bsz,
                                     gpu_id=settings_model.gpu_id, crit=critic, output_dir=save_in)
        dt.main(plot_examples=0, render_curves=False, model_name=detector.value, MAX_DISTANCE_OBJ=d,
                MAX_DISTANCE_INTERSECT=r, MAX_TIME_INTERSECT=t, recall_type="PRED AL NUMERATORE")
        return confidence_threshold, criticality_threshold, d, r, t

    def _compute_goal2(self):
        settings_model = self.settings_model
        detectors = settings_model.nuscenes_detectors
        path = settings_model.path_for_object_detectors_result_dir
        file_json = settings_model.path_for_object_detectors_result_json_file
        result_path = settings_model.results_path
        goal = settings_model.goal.value

        for detector_name, folder in detectors.items():
            # path + json file where detection results from mmdetection3d are stored, ready to be processed
            detector_file = path + folder + file_json
            # results of evaluation will be stored here
            output_folder = result_path + folder + "/"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            pkl_crit_results_store = self._iterate_drt_goal2(detector_name,
                                                             detector_file,
                                                             output_folder)

            self.file_util.write_data_to_file(output_folder_path=output_folder + 'pkl_crit_results_'+goal+'.json',
                                              key="PKL_CRIT_RESULTS_"+goal,
                                              value=pkl_crit_results_store)

    def _compute_goal1(self):
        settings_model = self.settings_model
        detectors = settings_model.nuscenes_detectors
        path = settings_model.path_for_object_detectors_result_dir
        file_json = settings_model.path_for_object_detectors_result_json_file
        result_path = settings_model.results_path

        # iterate on all detectors (SSN, POINTPILLARS, ...)
        for detector_name, folder in detectors.items():
            print(detector_name, folder)
            # path + json file where detection results from mmdetection3d are stored, ready to be processed
            detector_file = path + folder + file_json
            # results of evaluation will be stored here
            output_folder = result_path + folder + "/"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                # iterate on drt

            pkl_results_store, pkl_crit_results_store, ap_results_store = self._iterate_drt_goal1(detector_name,
                                                                                                  detector_file,
                                                                                                  output_folder)


            self.file_util.write_data_to_file(output_folder_path=output_folder + 'ap_results.json', key="AP_RESULTS",
                                              value=ap_results_store)
            self.file_util.write_data_to_file(output_folder_path=output_folder + 'pkl_results.json', key="PKL_RESULTS",
                                              value=pkl_results_store)
            self.file_util.write_data_to_file(output_folder_path=output_folder + 'pkl_crit_results_GOAL1.json',
                                              key="PKL_CRIT_RESULTS",
                                              value=pkl_crit_results_store)

    def _iterate_drt_goal1(self, detector_name, detector_file, output_folder):
        settings_model = self.settings_model
        dist_list = settings_model.dist
        goal = settings_model.goal
        drt = settings_model.drt
        verbose = settings_model.verbose
        nuscenes = settings_model.nuscenes
        conf_value = settings_model.conf_value
        object_classes = settings_model.arrayOfObjectClasses
        n_workers = settings_model.n_workers
        bsz = settings_model.bsz
        gpu_id = settings_model.gpu_id
        list_token = self.__list_token
        conf_th = settings_model.conf_th
        model_path = settings_model.pkl_planner_path
        mask_json = settings_model.mask_json_path
        criticalities = settings_model.criticalities
        save_in = settings_model.save_build_precision_recall_curve_data
        pkl_results_store = []
        pkl_crit_results_store = []
        ap_results_store = []
        for d, r, t in drt:
            print("DRT tuple is {}, {}, {}".format(d, r, t))
            print("Loading dt")
            dt = self.ColUtils.create_dt(detector_file,
                                         'val',
                                         model=detector_name,
                                         d=d,
                                         r=r,
                                         t=t,
                                         verbose=verbose,
                                         nuscenes=nuscenes,
                                         output_dir=save_in,
                                         conf_value=conf_value, n_workers=n_workers, bsz=bsz, gpu_id=gpu_id)
            # compute ap, ap_crit, and pkl, varying dist, confth
            print("Now evaluating object_classes {}".format(object_classes))
            ap_results, pkl_results, pkl_crit_results = self.ColUtils.compute_crit_pkl(dt=dt,
                                                                                       list_token=list_token,
                                                                                       conf_th_list=conf_th,
                                                                                       dist_list=dist_list,
                                                                                       criticalities=criticalities,
                                                                                       object_classes=object_classes,
                                                                                       verbose=verbose,
                                                                                       model_loaded=True,
                                                                                       model_object=self.ColUtils.load_pkl_model(
                                                                                           model_path, mask_json,
                                                                                           verbose), goal=goal)

            # storing values for later printing in json file
            ap_results.update({"DRT": {"D": d, "R": r, "T": t}})
            ap_results_store.append(ap_results)
            pkl_results_store.append(pkl_results)
            pkl_crit_results_store.append(pkl_crit_results)
            self.file_util.write_data_to_file(output_folder_path=output_folder + 'ap_results.json', key="AP_RESULTS",
                                              value=ap_results_store)
            self.file_util.write_data_to_file(output_folder_path=output_folder + 'pkl_results.json', key="PKL_RESULTS",
                                              value=pkl_results_store)
            self.file_util.write_data_to_file(output_folder_path=output_folder + 'pkl_crit_results_GOAL1.json',
                                              key="PKL_CRIT_RESULTS", value=pkl_crit_results_store)

            print("ONE FULL ROUND COMPLETED! drt= {}, model={}".format((d, r, t), detector_name))
            dt = None
        return pkl_results_store, pkl_crit_results_store, ap_results_store

    def _iterate_drt_goal2(self, detector_name, detector_file, output_folder):
        settings_model = self.settings_model
        dist_list = settings_model.dist
        goal = settings_model.goal
        drt = settings_model.drt
        verbose = settings_model.verbose
        nuscenes = settings_model.nuscenes
        conf_value = settings_model.conf_value
        object_classes = settings_model.arrayOfObjectClasses
        n_workers = settings_model.n_workers
        bsz = settings_model.bsz
        gpu_id = settings_model.gpu_id
        list_token = self.__list_token
        conf_th = settings_model.conf_th
        model_path = settings_model.pkl_planner_path
        mask_json = settings_model.mask_json_path
        criticalities = settings_model.criticalities
        save_in = settings_model.save_build_precision_recall_curve_data

        pkl_crit_results_store = []
        for d, r, t in drt:
            print("DRT tuple is {}, {}, {}".format(d, r, t))
            print("Loading dt")

            dt = self.ColUtils.create_dt(detector_file,
                                         'val',
                                         model=detector_name,
                                         d=d,
                                         r=r,
                                         t=t,
                                         verbose=verbose,
                                         output_dir=save_in,
                                         nuscenes=nuscenes,
                                         conf_value=conf_value, n_workers=n_workers, bsz=bsz, gpu_id=gpu_id)
            # compute ap, ap_crit, and pkl, varying dist, confth
            print("Now evaluating object_classes {}".format(object_classes))
            pkl_crit_results = self.ColUtils.compute_crit_pkl(dt=dt,
                                                              list_token=list_token,
                                                              conf_th_list=conf_th,
                                                              dist_list=dist_list,
                                                              criticalities=criticalities,
                                                              object_classes=object_classes,
                                                              verbose=verbose,

                                                              model_loaded=True,
                                                              model_object=self.ColUtils.load_pkl_model(
                                                                  model_path, mask_json,
                                                                  verbose), goal=goal)

            # storing values for later printing in json file
            pkl_crit_results_store.append(pkl_crit_results)
            self.file_util.write_data_to_file(output_folder_path=output_folder + 'pkl_crit_results_'+goal.value+'.json',
                                              key="PKL_CRIT_RESULTS_"+goal.value, value=pkl_crit_results_store)

            print("ONE FULL ROUND COMPLETED! drt= {}, model={}".format((d, r, t), detector_name))
            dt = None
        return pkl_crit_results_store
