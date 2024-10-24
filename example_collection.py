"""
       ['scene-0003', 'scene-0012', 'scene-0013', 'scene-0014', 'scene-0015', 'scene-0016', 'scene-0017', 'scene-0018',
         'scene-0035', 'scene-0036', 'scene-0038', 'scene-0039', 'scene-0092', 'scene-0093', 'scene-0094', 'scene-0095',
         'scene-0096', 'scene-0097', 'scene-0098', 'scene-0099', 'scene-0100', 'scene-0101', 'scene-0102', 'scene-0103',
         'scene-0104', 'scene-0105', 'scene-0106', 'scene-0107', 'scene-0108', 'scene-0109', 'scene-0110', 'scene-0221',
         'scene-0268', 'scene-0269', 'scene-0270', 'scene-0271', 'scene-0272', 'scene-0273', 'scene-0274', 'scene-0275',
         'scene-0276', 'scene-0277', 'scene-0278', 'scene-0329', 'scene-0330', 'scene-0331', 'scene-0332', 'scene-0344',
         'scene-0345', 'scene-0346', 'scene-0519', 'scene-0520', 'scene-0521', 'scene-0522', 'scene-0523', 'scene-0524',
         'scene-0552', 'scene-0553', 'scene-0554', 'scene-0555', 'scene-0556', 'scene-0557', 'scene-0558', 'scene-0559',
         'scene-0560', 'scene-0561', 'scene-0562', 'scene-0563', 'scene-0564', 'scene-0565', 'scene-0625', 'scene-0626',
         'scene-0627', 'scene-0629', 'scene-0630', 'scene-0632', 'scene-0633', 'scene-0634', 'scene-0635', 'scene-0636',
         'scene-0637', 'scene-0638', 'scene-0770', 'scene-0771', 'scene-0775', 'scene-0777', 'scene-0778', 'scene-0780',
         'scene-0781', 'scene-0782', 'scene-0783', 'scene-0784', 'scene-0794', 'scene-0795', 'scene-0796', 'scene-0797',
         'scene-0798', 'scene-0799', 'scene-0800', 'scene-0802', 'scene-0904', 'scene-0905', 'scene-0906', 'scene-0907',
         'scene-0908', 'scene-0909', 'scene-0910', 'scene-0911', 'scene-0912', 'scene-0913', 'scene-0914', 'scene-0915',
         'scene-0916', 'scene-0917', 'scene-0919', 'scene-0920', 'scene-0921', 'scene-0922', 'scene-0923', 'scene-0924',
         'scene-0925', 'scene-0926', 'scene-0927', 'scene-0928', 'scene-0929', 'scene-0930', 'scene-0931', 'scene-0962',
         'scene-0963', 'scene-0966', 'scene-0967', 'scene-0968', 'scene-0969', 'scene-0971', 'scene-0972', 'scene-1059',
         'scene-1060', 'scene-1061', 'scene-1062', 'scene-1063', 'scene-1064', 'scene-1065', 'scene-1066', 'scene-1067',
         'scene-1068', 'scene-1069', 'scene-1070', 'scene-1071', 'scene-1072', 'scene-1073']
"""

from model.DTOs.settings_model import SettingsModel
from src.enumerations import OBJECT_CLASSES, GOAL
import numpy as np
from enumerations import DETECTOR

from src.controller.collection_controller import CollectionController
from src.utils.file_utils import FileUtils
from src.utils.class_implementations.collection_utils import CollectionUtils
import json


# goal = GOAL.goal1
# notebook_home = "/Users/ibtihajnaeem/Documents/version_control/thesis/detectAndTrajectoryPackage/assets/"
# data_root = notebook_home + "nuscene/data"
# print(data_root)
# model_path = notebook_home + "pkl/planner.pt"
# mask_json = notebook_home + "pkl/masks_trainval.json"
# path = notebook_home + 'pkl/result_objects/'
# file_json = '/results_nusc.json'
# result_path = notebook_home + 'pkl/results/' + goal.name + '/retry_allobjects/'


# goal = GOAL.goal2
# notebook_home = "/home/notebook/"
# data_root = notebook_home + "nuscene/data"
# print(data_root)
# model_path = notebook_home + "pkl/Evaluation-of-Safety-Oriented-Metrics-for-Object-Detectors/metrics_model/planner.pt"
# mask_json = notebook_home + "pkl/Evaluation-of-Safety-Oriented-Metrics-for-Object-Detectors/metrics_model/masks_trainval.json"
# path = notebook_home + "pkl/result_objdet/"
# file_json = '/results_nusc.json'
# results_path = notebook_home + "ttpkl/results/" + goal.name + "/retry_allobjects/"


def compute_example():
    with open('json_file_for_example_collection.json', 'r') as f:
        config = json.load(f)
    goal = GOAL.goal1
    notebook_home = config["notebook_home"]
    data_root = config["data_root"]
    model_path = config["model_path"]
    mask_json = config["mask_json"]
    path = config["path"]
    file_json = config["file_json"]
    results_path = config["results_path"]
    scene_for_eval_set = config["scene_for_eval_set"]
    nuscenes_detectors = config["nuscenes_detectors"]
    number_of_image_bird_view = config["number_of_image_bird_view"]
    n_workers = config["n_workers"]
    bsz = config["bsz"]
    gpu_id = config["gpu_id"]
    mmdet3d_nuscenes_results_path = config["mmdet3d_nuscenes_results_path"]

    #array_of_object_classes = [OBJECT_CLASSES.car.value,OBJECT_CLASSES.truck.value]
    array_of_object_classes = [OBJECT_CLASSES.car.value, OBJECT_CLASSES.truck.value, OBJECT_CLASSES.bus.value,
                               OBJECT_CLASSES.trailer.value, OBJECT_CLASSES.construction_vehicle.value,
                               OBJECT_CLASSES.pedestrian.value, OBJECT_CLASSES.motorcycle.value,
                               OBJECT_CLASSES.bicycle.value, OBJECT_CLASSES.traffic_cone.value,
                               OBJECT_CLASSES.barrier.value]
    array_of_object_classes_reduced = [OBJECT_CLASSES.car.value]
    max_d = list(range(5, 55, 10))
    max_r = list(range(5, 55, 10))
    max_t = list(range(4, 25, 10))
    dist = [0.5, 1.0, 2.0, 4.0]
    conf_th = list(np.arange(0.05, 0.4, 0.05))
    criticalities = list(np.arange(0.10, 0.4, 0.05))
    settings = SettingsModel(
        results_path=results_path,
        detector={DETECTOR.POINTP.name: DETECTOR.POINTP.value},
        mmdet3d_nuscenes_results_path=mmdet3d_nuscenes_results_path,
        notebook_home=notebook_home,
        data_root=data_root,
        pkl_planner_path=model_path,
        mask_json_path=mask_json,
        path_for_object_detectors_result_dir=path,
        path_for_object_detectors_result_json_file=file_json,
        detector_file="none",
        goal=goal,
        array_of_object_classes=array_of_object_classes,
        array_of_object_classes_reduced=array_of_object_classes_reduced,
        max_d=max_d,
        max_r=max_r,
        max_t=max_t,
        verbose=False,
        dist=dist,
        conf_th=conf_th,
        criticalities=criticalities,
        n_workers=n_workers,
        bsz=bsz,
        gpu_id=gpu_id,
        number_of_image_bird_view=number_of_image_bird_view,
        nuscenes_detectors=nuscenes_detectors,
        scene_for_eval_set=scene_for_eval_set
    )
    file_utils = FileUtils()
    col_utils = CollectionUtils()
    collection_controller = CollectionController(settingsModel=settings, file_util=file_utils, ColUtils=col_utils)
    collection_controller.run()


compute_example()
