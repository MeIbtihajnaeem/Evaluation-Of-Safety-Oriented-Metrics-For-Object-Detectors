from typing import List
from enumerations import GOAL, OBJECT_CLASSES
from nuscenes import NuScenes
import nuscenes.eval.detection.config as config


class SettingsForGoal2Detector:
    def __init__(self,
                 mmdet3d_nuscenes_results_path: str, notebook_home: str,path_for_image_plots:str,
                 model_path: str, mask_json: str, path: str, file_json: str, data_root: str, result_path: str,
                 verbose=False, array_of_object_classes=None, d=10, r=10, t=4, crit=0.9, threshold=0.55,
                 n_workers=10, bsz=10,
                 gpu_id=0, number_of_image=0, nuscenes_detectors=None, scene_for_eval_set=None,detector=None,detector_file="none",number_of_trajectory_poses=16,trajectory_points = 20,font=8
                 ):
        if array_of_object_classes is None:
            array_of_object_classes = List[OBJECT_CLASSES.car]
        if nuscenes_detectors is None:
            nuscenes_detectors = {"SSN": 'SSN'}

        if not isinstance(data_root, str):
            raise ValueError(
                "Please provide a valid nuscene dataset install folder path")
        if not isinstance(notebook_home, str):
            raise ValueError(
                "Please provide a valid notebook home path ")
        if not isinstance(mmdet3d_nuscenes_results_path, str):
            raise ValueError(
                "Please provide a valid mmdet3d nuscenes results path ")
        if not isinstance(result_path, str):
            raise ValueError(
                "Please provide a valid result path of pkl results")
        if not isinstance(model_path, str):
            raise ValueError(
                "Please provide a valid path for planner.pt file")
        if not isinstance(mask_json, str):
            raise ValueError(
                "Please provide a valid path for masks_trainval.json file")
        if not isinstance(path, str):
            raise ValueError(
                "Please provide a valid path for results of the object detectors (result_nusc.json)")
        if not isinstance(file_json, str):
            raise ValueError(
                "object detectors (result_nusc.json)")
        if not isinstance(path_for_image_plots, str):
            raise ValueError(
                "Please provide a valid path for plots to be saved")
        if scene_for_eval_set is None:
            scene_for_eval_set = ['scene-0013',
                                  'scene-0554',
                                  'scene-0771',
                                  'scene-0929',
                                  'scene-1070',
                                  'scene-1072',
                                  'scene-0798',
                                  'scene-0108',
                                  'scene-0519',
                                  'scene-0332', ]

        self.goal = GOAL.goal2
        self.arrayOfObjectClasses = array_of_object_classes
        self.d = d
        self.r = r
        self.t = t
        self.crit = crit
        self.threshold = threshold
        self.n_workers = n_workers
        self.bsz = bsz
        self.gpu_id = gpu_id
        self.number_of_image = number_of_image
        self.nuscenes_detectors = nuscenes_detectors
        self.scene_for_eval_set = scene_for_eval_set
        self.nuscenes = NuScenes('v1.0-trainval', dataroot=data_root)
        self.conf_value = config.config_factory("detection_cvpr_2019")
        self.result_path = result_path
        self.model_path = model_path
        self.mask_json = mask_json
        self.verbose = verbose
        self.path = path
        self.file_json = file_json
        self.detector = detector
        self.detector_file = detector_file
        self.mmdet3d_nuscenes_results_path = mmdet3d_nuscenes_results_path
        self.notebook_home = notebook_home
        self.number_of_trajectory_poses = number_of_trajectory_poses
        self.trajectory_points = trajectory_points
        self.font = font
        self.path_for_image_plots = path_for_image_plots
