from ..model.settings_model import SettingsModel

import json
import src.utils.collection_utils as col_util
from enumerations import Goal
import os


class CollectionController:

    def __init__(self, settingsModel: SettingsModel):
        self.settingsModel = settingsModel
        self.__eval = settingsModel.scene_for_eval_set
        self.__val = settingsModel.scene_for_eval_set
        self.__scenes_list = col_util.get_scenes_list(settingsModel.nuscenes, self.settingsModel.scene_for_eval_set)
        self.__validation_samples = col_util.get_validation_samples(self.__scenes_list, settingsModel.nuscenes)
        self.__list_token = col_util.get_list_token(self.__validation_samples)
        with open(settingsModel.result_path + "/token_list.json", "w") as outfile:
            json.dump(self.__validation_samples, outfile)

    def _compute_goal1(self):
        settings_model = self.settingsModel
        dist_list = settings_model.dist
        detectors = settings_model.nuscenes_detectors
        goal = settings_model.goal
        path = settings_model.path_for_object_detectors_result_dir
        file_json = settings_model.path_for_object_detectors_result_json_file
        result_path = settings_model.result_path
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

        # iterate on all detectors (SSN, POINTPILLARS, ...)
        for detector_name, folder in detectors.items():
            if goal != Goal.goal1:  # just a bad trick to avoid this code if GOAL1 is not the target
                continue
            pkl_results_store = []
            pkl_crit_results_store = []
            ap_results_store = []
            print(detector_name, folder)
            # path + json file where detection results from mmdetection3d are stored, ready to be processed
            detector_file = path + folder + file_json
            # results of evaluation will be stored here
            output_folder = result_path + folder + "/"
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                # iterate on drt
            for d, r, t in drt:
                print("DRT tuple is {}, {}, {}".format(d, r, t))
                print("Loading dt")
                dt = col_util.create_dt(detector_file,
                                        'val',
                                        model=detector_name,
                                        d=d,
                                        r=r,
                                        t=t,
                                        verbose=verbose,
                                        nuscenes=nuscenes,
                                        conf_value=conf_value, n_workers=n_workers, bsz=bsz, gpu_id=gpu_id)
                # compute ap, ap_crit, and pkl, varying dist, confth
                print("Now evaluating object_classes {}".format(object_classes))
                ap_results, pkl_results, pkl_crit_results = col_util.compute_crit_pkl(dt=dt,
                                                                                      list_token=list_token,
                                                                                      conf_th_list=conf_th,
                                                                                      dist_list=dist_list,
                                                                                      criticalities=criticalities,
                                                                                      object_classes=object_classes,
                                                                                      verbose=verbose,
                                                                                      model_loaded=True,
                                                                                      model_object=col_util.load_pkl_model(
                                                                                          model_path, mask_json,
                                                                                          verbose),goal=goal)

                # storing values for later printing in json file
                ap_results.update({"DRT": {"D": d, "R": r, "T": t}})
                ap_results_store.append(ap_results)
                pkl_results_store.append(pkl_results)
                pkl_crit_results_store.append(pkl_crit_results)

                with open(output_folder + 'ap_results.json', 'w') as f:
                    json.dump({"AP_RESULTS": ap_results_store}, f)

                with open(output_folder + 'pkl_results.json', 'w') as f:
                    json.dump({"PKL_RESULTS": pkl_results_store}, f)

                with open(output_folder + 'pkl_crit_results_GOAL1.json', 'w') as f:
                    json.dump({"PKL_CRIT_RESULTS": pkl_crit_results_store}, f)

                print("ONE FULL ROUND COMPLETED! drt= {}, model={}".format((d, r, t), detector_name))
                dt = None

            with open(output_folder + 'ap_results.json', 'w') as f:
                json.dump({"AP_RESULTS": ap_results_store}, f)

            with open(output_folder + 'pkl_results.json', 'w') as f:
                json.dump({"PKL_RESULTS": pkl_results_store}, f)

            with open(output_folder + 'pkl_crit_results_GOAL1.json', 'w') as f:
                json.dump({"PKL_CRIT_RESULTS": pkl_crit_results_store}, f)
