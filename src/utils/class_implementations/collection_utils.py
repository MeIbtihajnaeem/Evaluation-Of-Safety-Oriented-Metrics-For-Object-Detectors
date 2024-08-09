from utils.modified_nuscenes_pkl.eval.detection.evaluate import DetectionEval
import torch
from planning_centric_metrics.models import compile_model
import os
import json
from src.enumerations import GOAL
from ..abstract_base.abstract_collection_utils import AbstractCollectionUtils
import matplotlib.pyplot as plt
import numpy as np


class CollectionUtils(AbstractCollectionUtils):

    def compute_crit_pkl(self, dt, criticalities, goal: GOAL,
                         list_token=None,
                         conf_th_list=None,
                         dist_list=None,
                         object_classes='car',
                         verbose=False,
                         model_loaded=False,  # an attempt to load the model outside the pkl
                         model_object=None, ):
        if list_token is None:
            list_token = []
        if conf_th_list is None:
            conf_th_list = [0.4]
        if dist_list is None:
            dist_list = [2.0]
        if goal == GOAL.goal1:
            results = dt.safety_metric_evaluation(
                list_of_tokens=list_token,
                conf_th_list=conf_th_list,
                dist_list=dist_list,
                crit_list=criticalities,
                obj_classes_list=object_classes,
                render_images=False,
                verbose=verbose,
                model_loaded=True,  # an attempt to load the model outside the pkl
                model_object=model_object,
            )
        else:
            results = dt.safety_metric_evaluation_GOAL2(
                list_of_tokens=list_token,
                conf_th_list=conf_th_list,
                dist_list=dist_list,
                crit_list=criticalities,
                obj_classes_list=object_classes,
                render_images=False,
                verbose=verbose,
                model_loaded=model_loaded,  # an attempt to load the model outside the pkl
                model_object=model_object,
            )

        return results

    def get_scenes_list(self, nuscenes, val):
        scenes_list = []
        counter = 0
        for i in nuscenes.scene:
            name = i['name']
            if name in val:
                counter = counter + 1
                scenes_list.append(i)
        return scenes_list

    def get_validation_samples(self, scenes_list, nuscenes):
        validation_samples = {}
        for i in scenes_list:
            scene_name = i['name']
            sample_token_list = []
            first_sample_token = i['first_sample_token']
            last_sample_token = i['last_sample_token']
            current_sample_token = first_sample_token
            sample_token_list.append(current_sample_token)
            if sample_token_list[0] != first_sample_token:
                print("error")
                break
            while current_sample_token != last_sample_token:
                sample = nuscenes.get('sample', current_sample_token)
                current_sample_token = sample['next']
                sample_token_list.append(current_sample_token)
            if sample_token_list[len(sample_token_list) - 1] != last_sample_token:
                print("error")
                break
            validation_samples.update({scene_name: sample_token_list})
        return validation_samples

    def get_list_token(self, validation_samples):
        list_token = []
        for i in validation_samples.keys():
            list_token.extend(validation_samples[i])
        return list_token

    def create_dt(self, detector_file=None, val=None, model=None, d=1, r=1, t=1, verbose=False, nuscenes=None,
                  conf_value=None,
                  n_workers=None, bsz=None, gpu_id=None, output_dir=None, crit=None):
        print("-----")
        print(output_dir)
        dt = DetectionEval(nusc=nuscenes,
                           config=conf_value,
                           result_path=detector_file,
                           eval_set=val,
                           model_name=model,
                           MAX_DISTANCE_OBJ=d,
                           MAX_DISTANCE_INTERSECT=r,
                           MAX_TIME_INTERSECT_OBJ=t,
                           verbose=verbose,
                           recall_type="PRED AL NUMERATORE",
                           nworkers=n_workers,
                           bsz=bsz,
                           gpuid=gpu_id,
                           output_dir=output_dir,
                           crit=crit
                           )
        return dt

    def load_pkl_model(self, model_path, mask_json, verbose):
        # constants related to how the planner was trained
        # layer_names = ['road_segment', 'lane']
        # line_names = ['road_divider', 'lane_divider']
        # stretch = 70.0

        device = torch.device(f'cuda:0')

        # load planner
        model = compile_model(cin=5, cout=16, with_skip=True,
                              dropout_p=0.0).to(device)

        if not os.path.isfile(model_path):
            print(f'downloading model weights to location {model_path}...')
            cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1feEIUjYSNWkl_b5SUkmPZ_-JAj3licJ9' -O {model_path}"
            print(f'running {cmd}')
            os.system(cmd)
            print(f'using model weights {model_path}')

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        model.to(device)
        model.eval()

        # load masks
        if not os.path.isfile(mask_json):
            print(f'downloading model masks to location {mask_json}...')
            cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=13M1xj9MkGo583ok9z8EkjQKSV8I2nWWF' -O {mask_json}"
            print(f'running {cmd}')
            os.system(cmd)
            if verbose:
                print(f'using location masks {mask_json}')
        with open(mask_json, 'r') as reader:
            mask_data = json.load(reader)
            masks = (torch.Tensor(mask_data) == 1).to(device)
        model_object = [model, masks]
        return model_object

    def plot_normal(self, data, confidence_threshold):
        detector = self.settings_model.detector
        save_in = self.settings_model.save_build_precision_recall_curve_data
        normal_05 = [data['car:0.5']['recall'], data['car:0.5']['precision'], data['car:0.5']['confidence']]
        normal_1 = [data['car:1.0']['recall'], data['car:1.0']['precision'], data['car:1.0']['confidence']]
        normal_2 = [data['car:2.0']['recall'], data['car:2.0']['precision'], data['car:2.0']['confidence']]
        normal_4 = [data['car:4.0']['recall'], data['car:4.0']['precision'], data['car:4.0']['confidence']]

        i, j, k = self._prc(normal_05, confidence_threshold)
        plt.plot(i, j, 'k', marker="o", markersize=5, markerfacecolor="black")
        plt.plot(normal_05[0], normal_05[1], 'k', linestyle='dotted', label='car:0.5')

        i, j, k = self._prc(normal_1, confidence_threshold)
        plt.plot(i, j, 'r', marker="o", markersize=5, markerfacecolor="red")
        plt.plot(normal_1[0], normal_1[1], 'r', linestyle='dashdot', label='car:1.0')

        i, j, k = self._prc(normal_2, confidence_threshold)
        plt.plot(i, j, 'g', marker="o", markersize=5, markerfacecolor="green")
        plt.plot(normal_2[0], normal_2[1], 'g', linestyle='dashed', label='car:2.0')

        i, j, k = self._prc(normal_4, confidence_threshold)
        plt.plot(i, j, 'y', marker="o", markersize=5, markerfacecolor="y")
        plt.plot(normal_4[0], normal_4[1], 'y', label='car:4.0')

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("DETECTOR: {}, marked conf.threshold {}".format(detector, confidence_threshold))
        plt.legend()
        plt.savefig(save_in + 'precision_recall.png')
        plt.show()
        return normal_05, normal_1, normal_2, normal_4

    def plot_special(self, data, confidence_threshold, save_in, d, r, t):
        detector = self.settings_model.detector
        save_in = self.settings_model.save_build_precision_recall_curve_data
        special_05 = [data['car:0.5']['recall'], data['car:0.5']['precision'], data['car:0.5']['confidence']]
        special_1 = [data['car:1.0']['recall'], data['car:1.0']['precision'], data['car:1.0']['confidence']]
        special_2 = [data['car:2.0']['recall'], data['car:2.0']['precision'], data['car:2.0']['confidence']]
        special_4 = [data['car:4.0']['recall'], data['car:4.0']['precision'], data['car:4.0']['confidence']]

        i, j, k = self._prc(special_05, confidence_threshold)
        plt.plot(i, j, 'k', marker="o", markersize=5, markerfacecolor="black")
        plt.plot(special_05[0], special_05[1], 'k', linestyle='dotted', label='car:0.5')

        i, j, k = self._prc(special_1, confidence_threshold)
        plt.plot(i, j, 'r', marker="o", markersize=5, markerfacecolor="r")
        plt.plot(special_1[0], special_1[1], 'r', linestyle='dashdot', label='car:1.0')

        i, j, k = self._prc(special_2, confidence_threshold)
        plt.plot(i, j, 'g', marker="o", markersize=5, markerfacecolor="g")
        plt.plot(special_2[0], special_2[1], 'g', linestyle='dashed', label='car:2.0')

        i, j, k = self._prc(special_4, confidence_threshold)
        plt.plot(i, j, 'y', marker="o", markersize=5, markerfacecolor="y")
        plt.plot(special_4[0], special_4[1], 'y', label='car:4.0')

        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title(
            "DETECTOR: {}, D={}, R={}, T={}, marked conf.threshold {}".format(detector, d, r, t, confidence_threshold))
        plt.legend()
        plt.savefig(save_in + 'precision_recall.png')
        plt.show()
        return special_05, special_1, special_2, special_4

    def plot_normal_and_special(self, special_05, special_1, special_2, special_4, normal_05, normal_1, normal_2,
                                normal_4, save_in, confidence_threshold, detector, d, r, t):

        # normal_precision, normal_recall, target_confidence = self._prc(normal_05, confidence_threshold)
        # special_precision, special_recall, special_confidence = self._prc(special_05, confidence_threshold)
        #
        # normal_precision, normal_recall, target_confidence = self._prc(normal_1, confidence_threshold)
        # special_precision, special_recall, special_confidence = self._prc(special_1, confidence_threshold)
        #
        # normal_precision, normal_recall, target_confidence = self._prc(normal_2, confidence_threshold)
        # special_precision, special_recall, special_confidence = self._prc(special_2, confidence_threshold)
        #
        # normal_precision, normal_recall, target_confidence = self._prc(normal_4, confidence_threshold)
        # special_precision, special_recall, special_confidence = self._prc(special_4, confidence_threshold)

        plt.plot(normal_05[0], np.asarray(normal_05[1]) - np.asarray(special_05[1]), 'k', linestyle='dotted',
                 label='car:0.5')
        plt.plot(normal_1[0], np.asarray(normal_1[1]) - np.asarray(special_1[1]), 'r', linestyle='dashdot',
                 label='car:1.0')
        plt.plot(normal_2[0], np.asarray(normal_2[1]) - np.asarray(special_2[1]), 'g', linestyle='dashed',
                 label='car:2.0')
        plt.plot(normal_4[0], np.asarray(normal_4[1]) - np.asarray(special_4[1]), 'y', label='car:4.0')

        plt.xlabel("recall levels")
        plt.ylabel("precision difference")
        plt.title("DETECTOR: {}, difference of precision normal vs modified bboxes".format(detector, d, r, t))
        plt.legend()
        plt.savefig(save_in + 'overlapped_figure.png')
        plt.show()

    def _prc(self, prec_rec_conf, target_confidence):
        list_confidences = np.asarray(prec_rec_conf[2])
        index = (np.abs(list_confidences - target_confidence)).argmin()
        # recall, precision, confidence
        return prec_rec_conf[0][index], prec_rec_conf[1][index], prec_rec_conf[2][index]
