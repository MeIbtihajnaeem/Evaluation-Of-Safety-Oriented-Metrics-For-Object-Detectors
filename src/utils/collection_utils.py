from .modified_nuscenes.eval.detection.evaluate import DetectionEval
import torch
from planning_centric_metrics.models import compile_model
import os
import json
from src.enumerations import Goal


def compute_crit_pkl(dt, criticalities, goal: Goal,
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
    if goal == Goal.goal1:
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


def get_scenes_list(nuscenes, val):
    scenes_list = []
    counter = 0
    for i in nuscenes.scene:
        name = i['name']
        if name in val:
            counter = counter + 1
            scenes_list.append(i)
    return scenes_list


def get_validation_samples(scenes_list, nuscenes):
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


def get_list_token(validation_samples):
    list_token = []
    for i in validation_samples.keys():
        list_token.extend(validation_samples[i])
    return list_token


def create_dt(detector_file=None, val=None, model=None, d=1, r=1, t=1, verbose=False,nuscenes=None,conf_value=None,n_workers=None,bsz=None,gpu_id =None):
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
                       gpuid=gpu_id
                       )
    return dt


def load_pkl_model(model_path, mask_json, verbose):
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
