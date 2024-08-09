# Copyright 2020 NVIDIA CORPORATION, Jonah Philion, Amlan Kar, Sanja Fidler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import json
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib as mpl

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from planning_centric_metrics import models
from planning_centric_metrics.planning_kl import EvalLoader
from planning_centric_metrics.planning_kl import make_rgba, render_observation

mpl.use('Agg')


def plot_heatmap(heat, masks):
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
              '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
              '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#808080', '#ffffff', '#000000']
    colors = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in colors]
    colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]
    plot_heat = heat.clone()
    plot_heat[~masks] = 0
    list_heats = []
    for ti in range(plot_heat.shape[0]):
        flat = plot_heat[ti].view(-1)
        ixes = flat.topk(20).indices
        flat[ixes] = 1
        flat = flat.view(plot_heat.shape[1], plot_heat.shape[2])
        showimg = make_rgba(np.clip(flat.numpy().T, 0, 1), colors[ti])
        plt.imshow(showimg, origin='lower')
        list_heats.append([flat, showimg])

    return list_heats


def plot_heatmap_GOAL3(heat, masks):
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
              '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
              '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
              '#000075', '#808080', '#ffffff', '#000000']
    colors = [tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for h in colors]
    colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]
    plot_heat = heat.detach().clone()
    plot_heat[~masks] = 0
    list_heats = []
    for ti in range(plot_heat.shape[0]):
        flat = plot_heat[ti].view(-1)
        ixes = flat.topk(20).indices
        flat[ixes] = 1
        flat = flat.view(plot_heat.shape[1], plot_heat.shape[2])
        showimg = make_rgba(np.clip(flat.numpy().T, 0, 1), colors[ti])
        # plt.imshow(showimg, origin='lower')
        list_heats.append([flat, showimg])

    return list_heats


def analyze_plot(gtxs, predxs, gtdist_sig, preddist_sig, masks, pkls=None, save_image=True):
    list_heatmap = []
    for i, (gtx, predx, gtsig, predsig) in enumerate(zip(gtxs, predxs,
                                                         gtdist_sig,
                                                         preddist_sig)):
        if save_image == True:
            fig = plt.figure(figsize=(9, 6))
        gs = mpl.gridspec.GridSpec(2, 3, left=0.01, bottom=0.01, right=0.99, top=0.99,
                                   wspace=0, hspace=0)
        if save_image == True:
            ax = plt.subplot(gs[0, 0])
            render_observation(gtx)
            ax.annotate("Ground Truth", xy=(0.05, 0.95), xycoords="axes fraction")
            ax = plt.subplot(gs[0, 1])
            render_observation(predx)
            ax.annotate("Detections", xy=(0.05, 0.95), xycoords="axes fraction")

        if (save_image == True):
            ax = plt.subplot(gs[0, 2])
        new_obs = gtx.clone()
        new_obs[3] = 0
        if (save_image == True):
            render_observation(new_obs)
            showimg = make_rgba(np.clip((-gtx[3] + predx[3]).numpy().T, 0, 1),
                                (1.0, 0.0, 0.0))
            plt.imshow(showimg, origin='lower')
            showimg = make_rgba(np.clip((gtx[3] - predx[3]).numpy().T, 0, 1),
                                (1.0, 0.0, 1.0))
            plt.imshow(showimg, origin='lower')
            plt.legend(handles=[
                mpatches.Patch(color=(1.0, 0.0, 0.0), label='False Positive'),
                mpatches.Patch(color=(1.0, 0.0, 1.0), label='False Negative'),
            ], loc='upper right')
            if pkls is not None:
                ax.annotate(f"PKL: {pkls[i]:.2f}", xy=(0.05, 0.95),
                            xycoords="axes fraction")
            ax = plt.subplot(gs[1, 0])

        heat_gt = plot_heatmap(gtsig, masks)
        if (save_image == True):
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax = plt.subplot(gs[1, 1])

        heat_pt = plot_heatmap(predsig, masks)
        list_heatmap.append({"heat_gt": heat_gt, "heat_pt": heat_pt, "i": i, "gtx": gtx, "ptx": predx})

        if (save_image == True):
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

            imname = f'worst{i:04}.jpg'
            print('saving', imname)
            plt.savefig(imname)
            plt.close(fig)
    return list_heatmap


def analyze_plot_GOAL3(gtxs, predxs, gtdist_sig, preddist_sig, masks, pkls=None, save_image=True):
    list_heatmap = []
    for i, (gtx, predx, gtsig, predsig) in enumerate(zip(gtxs, predxs,
                                                         gtdist_sig,
                                                         preddist_sig)):
        heat_gt = plot_heatmap_GOAL3(gtsig, masks)
        heat_pt = plot_heatmap_GOAL3(predsig, masks)
        list_heatmap.append({"heat_gt": heat_gt, "heat_pt": heat_pt, "i": i, "gtx": gtx, "ptx": predx})

    return list_heatmap


def calculate_pkl(gt_boxes, pred_boxes, sample_tokens, nusc,
                  nusc_maps, device, nworkers,
                  bsz=128, plot_kextremes=0, verbose=True,
                  modelpath='./planner.pt',
                  mask_json='./masks_trainval.json',
                  model_loaded=False,  # an attempt to load the model outside the pkl
                  model_object=None):
    r""" Computes the PKL https://arxiv.org/abs/2004.08745. It is designed to
    consume boxes in the format from
    nuscenes.eval.detection.evaluate.DetectionEval.
    Args:
            gt_boxes (EvalBoxes): Ground truth objects
            pred_boxes (EvalBoxes): Predicted objects
            sample_tokens List[str]: timestamps to be evaluated
            nusc (NuScenes): parser object provided by nuscenes-devkit
            nusc_maps (dict): maps map names to NuScenesMap objects
            device (torch.device): device for running forward pass
            nworkers (int): number of workers for dataloader
            bsz (int): batch size for dataloader
            plot_kextremes (int): number of examples to plot
            verbose (bool): print or not
            modelpath (str): File path to model weights.
                             Will download if not found.
            mask_json (str): File path to trajectory masks.
                             Will download if not found.
    Returns:
            info (dict) : dictionary of PKL scores
    """

    # constants related to how the planner was trained
    layer_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    stretch = 70.0

    if (model_loaded == False):
        # load planner
        print('loading pkl model')
        model = models.compile_model(cin=5, cout=16, with_skip=True,
                                     dropout_p=0.0).to(device)
        if not os.path.isfile(modelpath):
            print(f'downloading model weights to location {modelpath}...')
            cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1feEIUjYSNWkl_b5SUkmPZ_-JAj3licJ9' -O {modelpath}"
            print(f'running {cmd}')
            os.system(cmd)
        if verbose:
            print(f'using model weights {modelpath}')
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
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
            masks = (torch.Tensor(json.load(reader)) == 1).to(device)

    else:
        model = model_object[0]  # here is hte model
        masks = model_object[1]  # here is the mask

    dataset = EvalLoader(gt_boxes, pred_boxes, sample_tokens, nusc,
                         nusc_maps, stretch,
                         layer_names, line_names)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz,
                                             shuffle=False,
                                             num_workers=nworkers)

    if verbose:
        print('calculating pkl...')

    all_pkls = []
    for gtxs, predxs in dataloader:  # tqdm(dataloader):
        # print("size of dataloader is {}".format(len(dataloader.dataset)))
        with torch.no_grad():
            #            print("predxs.shape {} gtxs.shape {}".format(predxs.shape, gtxs.shape))
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))

        pkls = (F.binary_cross_entropy_with_logits(preddist[:, masks],
                                                   gtdist_sig[:, masks],
                                                   reduction='none')
                - F.binary_cross_entropy_with_logits(gtdist[:, masks],
                                                     gtdist_sig[:, masks],
                                                     reduction='none')).sum(1)

        all_pkls.append(pkls.cpu())
    all_pkls = torch.cat(all_pkls)
    print("pkls computed with output of size : {}".format(len(all_pkls)))

    # plot k extremes
    if verbose:
        print(f' calculate_pkl plotting {plot_kextremes} timestamps...')
    if plot_kextremes > 0:
        worst_ixes = all_pkls.topk(plot_kextremes).indices
        out = [dataset[i] for i in worst_ixes]
        gtxs, predxs = list(zip(*out))
        gtxs, predxs = torch.stack(gtxs), torch.stack(predxs)
        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))
        analyze_plot(gtxs, predxs, gtdist_sig.cpu(), preddist.sigmoid().cpu(),
                     masks.cpu(), pkls=all_pkls[worst_ixes])

    info = {
        'min': all_pkls.min().item(),
        'max': all_pkls.max().item(),
        'mean': all_pkls.mean().item(),
        'median': all_pkls.median().item(),
        'std': all_pkls.std().item(),
        'full': {tok: pk.item() for tok, pk in zip(sample_tokens, all_pkls)},
    }

    return info


def test_pkl(gt_boxes, pred_boxes, sample_tokens, nusc,
             nusc_maps, device, nworkers,
             bsz=128, plot_kextremes=0, verbose=True,
             modelpath='./planner.pt',
             mask_json='./masks_trainval.json', visualize_only=False,
             model_loaded=False,  # an attempt to load the model outside the pkl
             model_object=None):
    r""" Computes the PKL https://arxiv.org/abs/2004.08745. It is designed to
    consume boxes in the format from
    nuscenes.eval.detection.evaluate.DetectionEval.
    Args:
            gt_boxes (EvalBoxes): Ground truth objects
            pred_boxes (EvalBoxes): Predicted objects
            sample_tokens List[str]: timestamps to be evaluated
            nusc (NuScenes): parser object provided by nuscenes-devkit
            nusc_maps (dict): maps map names to NuScenesMap objects
            device (torch.device): device for running forward pass
            nworkers (int): number of workers for dataloader
            bsz (int): batch size for dataloader
            plot_kextremes (int): number of examples to plot
            verbose (bool): print or not
            modelpath (str): File path to model weights.
                             Will download if not found.
            mask_json (str): File path to trajectory masks.
                             Will download if not found.
    Returns:
            info (dict) : dictionary of PKL scores
    """

    # constants related to how the planner was trained
    layer_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    stretch = 70.0

    if (model_loaded == False):
        # load planner
        print('loading pkl model')
        model = models.compile_model(cin=5, cout=16, with_skip=True,
                                     dropout_p=0.0).to(device)
        if not os.path.isfile(modelpath):
            print(f'downloading model weights to location {modelpath}...')
            cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1feEIUjYSNWkl_b5SUkmPZ_-JAj3licJ9' -O {modelpath}"
            print(f'running {cmd}')
            os.system(cmd)
        if verbose:
            print(f'using model weights {modelpath}')
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
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
            masks = (torch.Tensor(json.load(reader)) == 1).to(device)

    else:
        model = model_object[0]  # here is hte model
        masks = model_object[1]  # here is the mask

    dataset = EvalLoader(gt_boxes, pred_boxes, sample_tokens, nusc,
                         nusc_maps, stretch,
                         layer_names, line_names)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz,
                                             shuffle=False,
                                             num_workers=nworkers)

    if verbose:
        print('calculating pkl...')

    all_pkls = []
    for gtxs, predxs in tqdm(dataloader):
        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))

        pkls = (F.binary_cross_entropy_with_logits(preddist[:, masks],
                                                   gtdist_sig[:, masks],
                                                   reduction='none')
                - F.binary_cross_entropy_with_logits(gtdist[:, masks],
                                                     gtdist_sig[:, masks],
                                                     reduction='none')).sum(1)

        all_pkls.append(pkls.cpu())
    all_pkls = torch.cat(all_pkls)

    # plot k extremes
    if verbose:
        print(f'plotting {plot_kextremes} timestamps...')
    if plot_kextremes > 0:
        worst_ixes = all_pkls.topk(plot_kextremes).indices
        print(worst_ixes)
        print(all_pkls.topk(plot_kextremes).indices)

        out = [dataset[i] for i in worst_ixes]
        gtxs, predxs = list(zip(*out))
        gtxs, predxs = torch.stack(gtxs), torch.stack(predxs)
        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))
        analyze_plot(gtxs, predxs, gtdist_sig.cpu(), preddist.sigmoid().cpu(),
                     masks.cpu(), pkls=all_pkls[worst_ixes])

    info = {
        'min': all_pkls.min().item(),
        'max': all_pkls.max().item(),
        'mean': all_pkls.mean().item(),
        'median': all_pkls.median().item(),
        'std': all_pkls.std().item(),
        'full': {tok: pk.item() for tok, pk in zip(sample_tokens, all_pkls)},
    }

    return info, all_pkls, gtdist, preddist, gtxs, predxs, worst_ixes


def test_pkl_2(gt_boxes, pred_boxes, sample_tokens, nusc,
               nusc_maps, device, nworkers,
               bsz=128, plot_kextremes=0, verbose=True,
               modelpath='./planner.pt',
               mask_json='./masks_trainval.json', visualize_only=False,
               model_loaded=False,  # an attempt to load the model outside the pkl
               model_object=None):
    r""" Computes the PKL https://arxiv.org/abs/2004.08745. It is designed to
    consume boxes in the format from
    nuscenes.eval.detection.evaluate.DetectionEval.
    Args:
            gt_boxes (EvalBoxes): Ground truth objects
            pred_boxes (EvalBoxes): Predicted objects
            sample_tokens List[str]: timestamps to be evaluated
            nusc (NuScenes): parser object provided by nuscenes-devkit
            nusc_maps (dict): maps map names to NuScenesMap objects
            device (torch.device): device for running forward pass
            nworkers (int): number of workers for dataloader
            bsz (int): batch size for dataloader
            plot_kextremes (int): number of examples to plot
            verbose (bool): print or not
            modelpath (str): File path to model weights.
                             Will download if not found.
            mask_json (str): File path to trajectory masks.
                             Will download if not found.
    Returns:
            info (dict) : dictionary of PKL scores
    """

    # constants related to how the planner was trained
    layer_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    stretch = 70.0

    if (model_loaded == False):
        # load planner
        print('loading pkl model')
        model = models.compile_model(cin=5, cout=16, with_skip=True,
                                     dropout_p=0.0).to(device)
        if not os.path.isfile(modelpath):
            print(f'downloading model weights to location {modelpath}...')
            cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1feEIUjYSNWkl_b5SUkmPZ_-JAj3licJ9' -O {modelpath}"
            print(f'running {cmd}')
            os.system(cmd)
        if verbose:
            print(f'using model weights {modelpath}')
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
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
            masks = (torch.Tensor(json.load(reader)) == 1).to(device)

    else:
        model = model_object[0]  # here is the model
        masks = model_object[1]  # here is the mask

    dataset = EvalLoader(gt_boxes, pred_boxes, sample_tokens, nusc,
                         nusc_maps, stretch,
                         layer_names, line_names)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz,
                                             shuffle=False,
                                             num_workers=nworkers)

    if verbose:
        print('calculating pkl...')

    all_pkls = []
    for gtxs, predxs in tqdm(dataloader):
        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))

        pkls = (F.binary_cross_entropy_with_logits(preddist[:, masks],
                                                   gtdist_sig[:, masks],
                                                   reduction='none')
                - F.binary_cross_entropy_with_logits(gtdist[:, masks],
                                                     gtdist_sig[:, masks],
                                                     reduction='none')).sum(1)

        all_pkls.append(pkls.cpu())
    all_pkls = torch.cat(all_pkls)
    index_list = None
    # plot k extremes
    #    if verbose:
    #        print(f'plotting {plot_kextremes} timestamps...')
    #    if plot_kextremes > 0:
    #        index_list = range(0, plot_kextremes)

    #        out = [dataset[i] for i in index_list ]
    #        gtxs, predxs = list(zip(*out))
    #        gtxs, predxs = torch.stack(gtxs), torch.stack(predxs)
    #        with torch.no_grad():
    #            gtdist = model(gtxs.to(device))
    #            gtdist_sig = gtdist.sigmoid()
    #            preddist = model(predxs.to(device))
    #        analyze_plot(gtxs, predxs, gtdist_sig.cpu(), preddist.sigmoid().cpu(),
    #                     masks.cpu(), pkls=all_pkls[index_list])

    info = {
        'min': all_pkls.min().item(),
        'max': all_pkls.max().item(),
        'mean': all_pkls.mean().item(),
        'median': all_pkls.median().item(),
        'std': all_pkls.std().item(),
        'full': {tok: pk.item() for tok, pk in zip(sample_tokens, all_pkls)},
    }

    return info, all_pkls, gtdist, preddist, gtxs, predxs, index_list, gt_boxes, nusc, nusc_maps, dataset


# used only in goal3
def pkl_print_visualizations(gt_boxes, pred_boxes, sample_tokens, nusc,
                             nusc_maps, device, nworkers,
                             bsz=128, plot_kextremes=0, verbose=True,
                             modelpath='./planner.pt',
                             mask_json='./masks_trainval.json', visualize_only=False,
                             model_loaded=False,  # an attempt to load the model outside the pkl
                             model_object=None):
    # constants related to how the planner was trained
    layer_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    stretch = 70.0

    if model_loaded == False:
        # load planner
        print('loading pkl model')
        model = models.compile_model(cin=5, cout=16, with_skip=True,
                                     dropout_p=0.0).to(device)
        if not os.path.isfile(modelpath):
            print(f'downloading model weights to location {modelpath}...')
            cmd = f"wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1feEIUjYSNWkl_b5SUkmPZ_-JAj3licJ9' -O {modelpath}"
            print(f'running {cmd}')
            os.system(cmd)
        if verbose:
            print(f'using model weights {modelpath}')
        model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
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
            masks = (torch.Tensor(json.load(reader)) == 1).to(device)

    else:
        model = model_object[0]  # here is the model
        masks = model_object[1]  # here is the mask

    dataset = EvalLoader(gt_boxes, pred_boxes, sample_tokens, nusc,
                         nusc_maps, stretch,
                         layer_names, line_names)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsz,
                                             shuffle=False,
                                             num_workers=nworkers)

    if verbose:
        print('calculating pkl...')

    all_pkls = []
    for gtxs, predxs in dataloader:
        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))

        pkls = (F.binary_cross_entropy_with_logits(preddist[:, masks],
                                                   gtdist_sig[:, masks],
                                                   reduction='none')
                - F.binary_cross_entropy_with_logits(gtdist[:, masks],
                                                     gtdist_sig[:, masks],
                                                     reduction='none')).sum(1)

        all_pkls.append(pkls.cpu())
    all_pkls = torch.cat(all_pkls)
    if verbose:
        print("pkls computed with output of size: {}".format(len(all_pkls)))
        print("all_pkls: {}".format((all_pkls)))

    # plot k extremes
    if verbose:
        print(f'plotting {plot_kextremes} timestamps...')

    if plot_kextremes > 0:  # needed, to create prediction arrays
        # indexes = all_pkls.topk(plot_kextremes).indices
        out = [dataset[i] for i in range(0, plot_kextremes)]
        gtxs, predxs = list(zip(*out))
        gtxs, predxs = torch.stack(gtxs), torch.stack(predxs)
        with torch.no_grad():
            gtdist = model(gtxs.to(device))
            gtdist_sig = gtdist.sigmoid()
            preddist = model(predxs.to(device))
        createdimages = analyze_plot_GOAL3(gtxs, predxs, gtdist_sig.cpu(), preddist.sigmoid().cpu(), masks.cpu(),
                                           pkls=all_pkls, save_image=False)
    info = {
        'min': all_pkls.min().item(),
        'max': all_pkls.max().item(),
        'mean': all_pkls.mean().item(),
        'median': all_pkls.median().item(),
        'std': all_pkls.std().item(),
        'full': {tok: pk.item() for tok, pk in zip(sample_tokens, all_pkls)},
    }

    #    gtxs.cpu()
    #    model.cpu()
    #    masks.cpu()
    #    del model
    #    del out
    #    del pkls
    #    del dataloader
    #    del dataset
    return info, all_pkls, gtdist, preddist, gtxs, predxs, createdimages, gt_boxes, nusc_maps
