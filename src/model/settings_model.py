from ..enumerations import Goal, ObjectClasses
from typing import List
import itertools
from nuscenes import NuScenes

# custom imports
import nuscenes.eval.detection.config as config


class SettingsModel:
    def __init__(self,
                 mmdet3d_nuscenes_results_path: str,
                 notebook_home: str,
                 data_root: str,
                 pkl_planner_path: str,
                 mask_json_path: str,
                 path_for_object_detectors_result_dir: str,
                 path_for_object_detectors_result_json_file: str,
                 goal=None,
                 array_of_object_classes=None,
                 array_of_object_classes_reduced=None,
                 max_d=None,
                 max_r=None,
                 max_t=None,
                 verbose = None,
                 dist=None,
                 conf_th=None,
                 criticalities=None,
                 n_workers=10,
                 bsz=128,
                 gpu_id=0,
                 number_of_image_bird_view=0,
                 nuscenes_detectors=None,
                 scene_for_eval_set=None):
        if max_d is None:
            max_d = [15, 20, 25]
        if max_d is None:
            max_r = [15, 20, 25]
        if max_t is None:
            max_r = [12, 16, 20]

        if array_of_object_classes is None:
            array_of_object_classes = List[ObjectClasses.car]
        if array_of_object_classes_reduced is None:
            array_of_object_classes_reduced = List[ObjectClasses.car]
        if goal is None:
            goal = Goal.goal1
        if dist is None:
            dist = [0.5, 1.0, 2.0, 4.0]
        if conf_th is None:
            conf_th = [0.4, 0.45, 0.5]
        if criticalities is None:
            criticalities = [0.4, 0.5, 0.6, 0.7, 0.8]
        if nuscenes_detectors is None:
            nuscenes_detectors = {"PointPillars": 'POINTP', }
        if scene_for_eval_set is None:
            scene_for_eval_set = ['scene-0519', 'scene-0013', ]
        if verbose is None:
            verbose = False

        if not isinstance(mmdet3d_nuscenes_results_path, str):
            raise ValueError(
                "Please provide a valid mmdetection3d result path")
        if not isinstance(notebook_home, str):
            raise ValueError(
                "Please provide a valid notebook home path")
        if not isinstance(data_root, str):
            raise ValueError(
                "Please provide a valid nuscene dataset install folder path")
        if not isinstance(pkl_planner_path, str):
            raise ValueError(
                "Please provide a valid pkl planner.pt file path")
        if not isinstance(mask_json_path, str):
            raise ValueError(
                "Please provide a valid trainval.json file path")
        if not isinstance(path_for_object_detectors_result_dir, str):
            raise ValueError(
                "Please provide a valid path for results of the object detectors directory")
        if not isinstance(path_for_object_detectors_result_json_file, str):
            raise ValueError(
                "Please provide a valid path for results of the object detectors json file (result_nusc.json)")

        self.mmdet3d_nuscenes_results_path = mmdet3d_nuscenes_results_path
        self.notebook_home = notebook_home

        self.goal = goal
        self.arrayOfObjectClasses = array_of_object_classes
        self.array_of_object_classes_reduced = array_of_object_classes_reduced
        self.maxD = max_d
        self.maxR = max_r
        self.maxT = max_t
        self.dist = dist
        self.conf_th = conf_th
        self.criticalities = criticalities
        self.n_workers = n_workers
        self.bsz = bsz
        self.gpu_id = gpu_id
        self.number_of_image_bird_view = number_of_image_bird_view
        self.nuscenes_detectors = nuscenes_detectors
        self.scene_for_eval_set = scene_for_eval_set
        self.data_root = data_root
        self.result_path = notebook_home + "'pkl/results/GOAL2/retry_allobjects/"
        self.drt = list(itertools.product(*[max_d, max_r, max_t]))
        self.nuscenes = NuScenes('v1.0-trainval', dataroot=data_root)
        self.conf_value = config.config_factory("detection_cvpr_2019")
        self.verbose = verbose
        self.path_for_object_detectors_result_dir = path_for_object_detectors_result_dir
        self.path_for_object_detectors_result_json_file = path_for_object_detectors_result_json_file
        self.pkl_planner_path = pkl_planner_path
        self.mask_json_path = mask_json_path

    def __repr__(self):
        return f"SettingsModel(goal={self.goal}, mmdet3d_nuscenes_results_path='{self.mmdet3d_nuscenes_results_path}')"

    def to_dict(self):
        return {
            "goal": self.goal,
            "mmdet3d_nuscenes_results_path": self.mmdet3d_nuscenes_results_path
        }

    @classmethod
    def from_dict(cls, data):
        goal = data.get("goal")
        mmdet3d_nuscenes_results_path = data.get("mmdet3d_nuscenes_results_path")

        return cls(goal=goal, mmdet3d_nuscenes_results_path=mmdet3d_nuscenes_results_path)
