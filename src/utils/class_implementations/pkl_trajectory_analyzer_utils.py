from model.DTOs.settings_model_for_goal2_detector import SettingsForGoal2Detector
import numpy as np
import matplotlib.pyplot as plt
from model.Domain_models.trajectory_model import TrajectoryModel


class PKLTrajectoryAnalyzer:
    @staticmethod
    def process_all_tokens(settingsModel: SettingsForGoal2Detector, token_list_in_use, dt):
        number_of_trajectory_poses = settingsModel.number_of_trajectory_poses  # 4 seconds prediction
        trajectory_points = settingsModel.trajectory_points  # for every frame, we pick the top 20 trajectory points. This 20 comes from pkl functions.
        threshold = settingsModel.threshold
        crit = settingsModel.crit
        object_classes = settingsModel.arrayOfObjectClasses
        verbose = settingsModel.verbose
        result_path = settingsModel.result_path
        # %matplotlib
        # inline
        font = 8

        for token in token_list_in_use:
            print("elaborating token: {}".format(token))
            pklfile_dynamic, info_dynamic, all_pkls_dynamic, \
                gtdist, preddist_dynamic, gtxs, \
                predxs_dynamic, createdimages_dynamic = dt.calc_sample_crit_GOAL2_unique_param(listtoken=[token],
                                                                                               conf_th=threshold,
                                                                                               crit=crit,
                                                                                               obj_classes_list=object_classes,
                                                                                               # filter boxes based on class
                                                                                               verbose=verbose)
            pklfile_original, info_original, all_pkls_original, gtdist, preddist_original, gtxs, \
                predxs_original, \
                createdimages_original = dt.calc_sample_crit_GOAL2_unique_param(listtoken=[token],
                                                                                conf_th=threshold,
                                                                                crit=10.00,
                                                                                obj_classes_list=object_classes,
                                                                                # filter boxes based on class
                                                                                verbose=verbose)

            print("pkl original: {} pkl dynamic: {}".format(info_original['mean'], info_dynamic['mean']))
            # get the ground truth bboxes
            sGT = createdimages_original[0]['heat_gt']
            ground_truths = createdimages_original[0]['gtx'].cpu().numpy()[3]
            ground_truths = np.swapaxes(ground_truths, 0, 1)

            # get the ground truth trajectory
            dimension = sGT[0][0].cpu().numpy().shape
            basic_shape = np.zeros(dimension)
            trajectory_gt = basic_shape
            for i in range(0, number_of_trajectory_poses):
                # get 20 most relevant points in sGT[i][0].cpu().numpy(), i.e., per frame
                tmp = sGT[i][0].cpu().numpy()
                tmp_most_relevant_index = np.argsort(tmp.flatten())[-trajectory_points:]
                tmp.fill(0)
                flat = tmp.flatten()
                flat[tmp_most_relevant_index] = 1
                tmp = np.reshape(flat, dimension)
                trajectory_gt = np.add(trajectory_gt, tmp)

            trajectory_gt[trajectory_gt > 0] = 1
            # trajectory_gt[trajectory_gt<THRESHOLD]=0
            trajectory_gt = np.swapaxes(trajectory_gt, 0, 1)

            # get the predicted original bboxes
            sPT = createdimages_original[0]['heat_pt']
            original_detections = createdimages_original[0]['ptx'].cpu().numpy()[3]
            original_detections = np.swapaxes(original_detections, 0, 1)

            # get the predicted original trajectories
            trajectory_pt = basic_shape
            for i in range(0, number_of_trajectory_poses):
                # get 20 most relevant points in sGT[i][0].cpu().numpy(), i.e., per frame
                tmp = sPT[i][0].cpu().numpy()
                tmp_most_relevant_index = np.argsort(tmp.flatten())[-trajectory_points:]
                tmp.fill(0)
                flat = tmp.flatten()
                flat[tmp_most_relevant_index] = 1
                tmp = np.reshape(flat, dimension)
                trajectory_pt = np.add(trajectory_pt, tmp)

            trajectory_pt[trajectory_pt > 0] = 1
            # trajectory_pt[trajectory_pt<THRESHOLD]=0
            trajectory_pt = np.swapaxes(trajectory_pt, 0, 1)

            # ground truth with dynamic object - just for check
            sDT_GT = createdimages_dynamic[0]['heat_gt']
            ground_truths_dt = createdimages_dynamic[0]['gtx'].cpu().numpy()[3]
            ground_truths_dt = np.swapaxes(ground_truths_dt, 0, 1)

            trajectory_gt_dt = basic_shape
            for i in range(0, number_of_trajectory_poses):
                # trajectory_gt_dt=np.add(trajectory_gt_dt,sDT_GT[i][0].cpu().numpy())
                # get 20 most relevant points in sGT[i][0].cpu().numpy(), i.e., per frame
                tmp = sDT_GT[i][0].cpu().numpy()
                tmp_most_relevant_index = np.argsort(tmp.flatten())[-trajectory_points:]
                tmp.fill(0)
                flat = tmp.flatten()
                flat[tmp_most_relevant_index] = 1
                tmp = np.reshape(flat, dimension)
                trajectory_gt_dt = np.add(trajectory_gt_dt, tmp)

            trajectory_gt_dt[trajectory_gt_dt > 0] = 1
            # trajectory_gt_dt[trajectory_gt_dt<THRESHOLD]=0
            trajectory_gt_dt = np.swapaxes(trajectory_gt_dt, 0, 1)

            # get the predicted dynamic (our approach) bboxes
            sDT = createdimages_dynamic[0]['heat_pt']
            dynamic_detections = createdimages_dynamic[0]['ptx'].cpu().numpy()[3]
            dynamic_detections = np.swapaxes(dynamic_detections, 0, 1)

            # get the predicted dynamic (our approach) trajectories
            trajectory_dt = basic_shape
            for i in range(0, number_of_trajectory_poses):
                # trajectory_dt=np.add(trajectory_dt,sDT[i][0].cpu().numpy())
                # get 20 most relevant points in sGT[i][0].cpu().numpy(), i.e., per frame
                tmp = sDT[i][0].cpu().numpy()
                tmp_most_relevant_index = np.argsort(tmp.flatten())[-trajectory_points:]
                tmp.fill(0)
                flat = tmp.flatten()
                flat[tmp_most_relevant_index] = 1
                tmp = np.reshape(flat, dimension)
                trajectory_dt = np.add(trajectory_dt, tmp)

            trajectory_dt[trajectory_dt > 0] = 1
            # trajectory_dt[trajectory_dt<THRESHOLD]=0
            trajectory_dt = np.swapaxes(trajectory_dt, 0, 1)

            # overlap with distance
            # ground truth
            final_gt = np.add(trajectory_gt, ground_truths)
            if 2 in final_gt:
                print("collision with trajectory ground truth and ground truth bboxes for token {}".format(token))
                with open(result_path + 'GOAL2.txt', 'a+') as f:
                    f.write(
                        "collision with trajectory ground truth and ground truth bboxes for token {}\n".format(token))
            # original
            final_pt = np.add(trajectory_pt, ground_truths)
            if 2 in final_pt:
                print("collision with trajectory original and ground truth for token {}".format(token))
                with open(result_path + 'GOAL2.txt', 'a+') as f:
                    f.write("collision with trajectory original and ground truth for token {}\n".format(token))
            # ground truth with dynamic object -- just for test
            final_gt_dt = np.add(trajectory_gt_dt, ground_truths)
            #    if 2 in final_gt_dt:
            #        print("collision with trajectory dynamic -our approach- and ground truth bboxes for token {}".format(token))
            #        with open(RESULTS_PATH+'GOAL2.txt', 'a+') as f:
            #            f.write("collision with trajectory dynamic -our approach- and ground truth bboxes for token {}\n".format(token))

            # dynamic
            final_dt = np.add(trajectory_dt, ground_truths)
            if 2 in final_dt:
                print("collision with trajectory dynamic -our approach- and ground truth bboxes for token {}".format(
                    token))
                with open(result_path + 'GOAL2.txt', 'a+') as f:
                    f.write(
                        "collision with trajectory dynamic -our approach- and ground truth bboxes for token {}\n".format(
                            token))

            print("printing figures...")
            return TrajectoryModel(font=font, trajectory_gt=trajectory_gt, trajectory_pt=trajectory_pt,
                                   trajectory_gt_dt=trajectory_gt_dt, final_gt_dt=final_gt_dt,
                                   ground_truths_dt=ground_truths_dt, trajectory_dt=trajectory_dt,
                                   ground_truths=ground_truths, final_pt=final_pt,
                                   dynamic_detections=dynamic_detections, final_gt=final_gt, final_dt=final_dt,
                                   original_detections=original_detections)

    @staticmethod
    def plot_trajectory(trajectory_model: TrajectoryModel, path_for_image_plots):

        font = trajectory_model.get_font()
        trajectory_gt = trajectory_model.get_trajectory_gt()
        trajectory_pt = trajectory_model.get_trajectory_pt()
        trajectory_gt_dt = trajectory_model.get_trajectory_gt_dt()
        final_gt_dt = trajectory_model.get_final_gt_dt()
        ground_truths_dt = trajectory_model.get_ground_truths_dt()
        trajectory_dt = trajectory_model.get_trajectory_dt()
        ground_truths = trajectory_model.get_ground_truths()
        final_pt = trajectory_model.get_final_pt()
        dynamic_detections = trajectory_model.get_dynamic_detections()
        final_gt = trajectory_model.get_final_dt()
        final_dt = trajectory_model.get_final_dt()
        original_detections = trajectory_model.get_original_detections()
        plt.clf()
        plt.figure()
        fig, (ax) = plt.subplots(3, 3)
        ax[0, 0].tick_params(axis='both', labelsize=font)
        ax[0, 1].tick_params(axis='both', labelsize=font)
        ax[0, 2].tick_params(axis='both', labelsize=font)
        ax[1, 0].tick_params(axis='both', labelsize=font)
        ax[1, 1].tick_params(axis='both', labelsize=font)
        ax[1, 2].tick_params(axis='both', labelsize=font)
        ax[2, 0].tick_params(axis='both', labelsize=font)
        ax[2, 1].tick_params(axis='both', labelsize=font)
        ax[2, 2].tick_params(axis='both', labelsize=font)
        fig.tight_layout()

        ax[0, 0].set_title('GT Traj', fontsize=font)
        ax[0, 0].imshow(trajectory_gt)

        ax[0, 1].set_title('Original Traj', fontsize=font)
        ax[0, 1].imshow(trajectory_pt)

        ax[0, 2].set_title('Dynamic Traj', fontsize=font)
        ax[0, 2].imshow(trajectory_dt)

        ax[1, 0].set_title('GT BBox and GT Traj', fontsize=font)
        image_gt = np.swapaxes(np.swapaxes(np.asarray([trajectory_gt, ground_truths, final_gt * 0]), 0, 2), 0, 1)
        ax[1, 0].imshow(1 - image_gt)

        ax[1, 1].set_title('GT BBox and Original Traj', fontsize=font)
        image_pt = np.swapaxes(np.swapaxes(np.asarray([trajectory_pt, ground_truths, final_pt * 0]), 0, 2), 0, 1)
        ax[1, 1].imshow(1 - image_pt)

        ax[1, 2].set_title('GT BBox and Dynamic Traj', fontsize=font)
        image_dt = np.swapaxes(np.swapaxes(np.asarray([trajectory_dt, ground_truths, final_dt * 0]), 0, 2), 0, 1)
        ax[1, 2].imshow(1 - image_dt)

        ax[2, 0].set_title('should be equal to above', fontsize=font)
        image_gt_dt = np.swapaxes(
            np.swapaxes(np.asarray([trajectory_gt_dt, ground_truths_dt, final_gt_dt * 0]), 0, 2), 0, 1)
        ax[2, 0].imshow(1 - image_gt_dt)

        ax[2, 1].set_title('Orig BBox and orig pkl', fontsize=font)
        image_pt = np.swapaxes(np.swapaxes(np.asarray([trajectory_pt, original_detections, final_pt * 0]), 0, 2), 0,
                               1)
        ax[2, 1].imshow(1 - image_pt)

        ax[2, 2].set_title('Dyn BBox and dyn pkl', fontsize=font)
        image_dt = np.swapaxes(np.swapaxes(np.asarray([trajectory_dt, dynamic_detections, final_dt * 0]), 0, 2), 0,
                               1)
        ax[2, 2].imshow(1 - image_dt)
        plt.savefig(path_for_image_plots + 'plot_trajectory_plot_trajectory.png')
        plt.show()
        print("round completed")

    @staticmethod
    def visualize_trajectory_and_detections(createdimages_dynamic, info_original, createdimages_original, info_dynamic,
                                            path_for_image_plots):
        # %matplotlib
        # inline
        font = 8
        number_of_trajectory_poses = 16  # 4 seconds prediction
        trajectory_points = 20  # for every frame, we pick the top 20 trajectory points. This 20 comes from pkl functions.

        print("pkl original: {} pkl dynamic: {}".format(info_original['mean'], info_dynamic['mean']))
        # get the ground truth bboxes
        sGT = createdimages_original[0]['heat_gt']
        ground_truths = createdimages_original[0]['gtx'].cpu().numpy()[3]
        ground_truths = np.swapaxes(ground_truths, 0, 1)

        # get the ground truth trajectory
        dimension = sGT[0][0].cpu().numpy().shape
        basic_shape = np.zeros(dimension)
        trajectory_gt = basic_shape
        for i in range(0, number_of_trajectory_poses):
            # get 20 most relevant points in sGT[i][0].cpu().numpy(), i.e., per frame
            tmp = sGT[i][0].cpu().numpy()
            tmp_most_relevant_index = np.argsort(tmp.flatten())[-trajectory_points:]
            tmp.fill(0)
            flat = tmp.flatten()
            flat[tmp_most_relevant_index] = 1
            tmp = np.reshape(flat, dimension)
            trajectory_gt = np.add(trajectory_gt, tmp)

        trajectory_gt[trajectory_gt > 0] = 1
        # trajectory_gt[trajectory_gt<THRESHOLD]=0
        trajectory_gt = np.swapaxes(trajectory_gt, 0, 1)

        # get the predicted original bboxes
        sPT = createdimages_original[0]['heat_pt']
        original_detections = createdimages_original[0]['ptx'].cpu().numpy()[3]
        original_detections = np.swapaxes(original_detections, 0, 1)

        # get the predicted original trajectories
        trajectory_pt = basic_shape
        for i in range(0, number_of_trajectory_poses):
            # get 20 most relevant points in sGT[i][0].cpu().numpy(), i.e., per frame
            tmp = sPT[i][0].cpu().numpy()
            tmp_most_relevant_index = np.argsort(tmp.flatten())[-trajectory_points:]
            tmp.fill(0)
            flat = tmp.flatten()
            flat[tmp_most_relevant_index] = 1
            tmp = np.reshape(flat, dimension)
            trajectory_pt = np.add(trajectory_pt, tmp)

        trajectory_pt[trajectory_pt > 0] = 1
        trajectory_pt = np.swapaxes(trajectory_pt, 0, 1)

        # get the predicted dynamic (our approach) bboxes
        sDT = createdimages_dynamic[0]['heat_pt']
        dynamic_detections = createdimages_dynamic[0]['ptx'].cpu().numpy()[3]
        dynamic_detections = np.swapaxes(dynamic_detections, 0, 1)

        # get the predicted dynamic (our approach) trajectories
        trajectory_dt = basic_shape
        for i in range(0, number_of_trajectory_poses):
            # get 20 most relevant points in sGT[i][0].cpu().numpy(), i.e., per frame
            tmp = sDT[i][0].cpu().numpy()
            tmp_most_relevant_index = np.argsort(tmp.flatten())[-trajectory_points:]
            tmp.fill(0)
            flat = tmp.flatten()
            flat[tmp_most_relevant_index] = 1
            tmp = np.reshape(flat, dimension)
            trajectory_dt = np.add(trajectory_dt, tmp)

        trajectory_dt[trajectory_dt > 0] = 1
        trajectory_dt = np.swapaxes(trajectory_dt, 0, 1)

        # overlap with distance
        # ground truth
        final_gt = np.add(trajectory_gt, ground_truths)

        # original
        final_pt = np.add(trajectory_pt, ground_truths)

        # dynamic
        final_dt = np.add(trajectory_dt, ground_truths)

        print("printing figures...")
        plt.clf()
        plt.figure()
        fig, (ax) = plt.subplots(2, 3)
        ax[0, 0].tick_params(axis='both', labelsize=font)
        ax[0, 1].tick_params(axis='both', labelsize=font)
        ax[0, 2].tick_params(axis='both', labelsize=font)
        ax[1, 0].tick_params(axis='both', labelsize=font)
        ax[1, 1].tick_params(axis='both', labelsize=font)
        ax[1, 2].tick_params(axis='both', labelsize=font)
        fig.tight_layout()

        ax[0, 0].set_title('GT Traj', fontsize=font)
        ax[0, 0].imshow(trajectory_gt)

        ax[0, 1].set_title('Original Traj', fontsize=font)
        ax[0, 1].imshow(trajectory_pt)

        ax[0, 2].set_title('Dynamic Traj', fontsize=font)
        ax[0, 2].imshow(trajectory_dt)

        ax[1, 0].set_title('GT BBox and GT Traj', fontsize=font)
        image_gt = np.swapaxes(np.swapaxes(np.asarray([trajectory_gt, ground_truths, final_gt * 0]), 0, 2), 0, 1)
        ax[1, 0].imshow(1 - image_gt)

        ax[1, 1].set_title('Orig BBox and orig pkl', fontsize=font)
        image_pt = np.swapaxes(np.swapaxes(np.asarray([trajectory_pt, original_detections, final_pt * 0]), 0, 2), 0, 1)
        ax[1, 1].imshow(1 - image_pt)

        ax[1, 2].set_title('Dyn BBox and dyn pkl', fontsize=font)
        image_dt = np.swapaxes(np.swapaxes(np.asarray([trajectory_dt, dynamic_detections, final_dt * 0]), 0, 2), 0, 1)
        ax[1, 2].imshow(1 - image_dt)
        plt.savefig(path_for_image_plots + 'visualize_trajectory_and_detections_visualize_trajectory_and_detections.png')
        plt.show()
        return trajectory_gt, trajectory_pt

    @staticmethod
    def render_road_and_borders(path_for_image_plots, gtxs):
        light_grey = 0.8

        # road is white with light gray borders
        road = np.swapaxes(gtxs[0][0], 0, 1)
        road[road != 1] = light_grey
        plt.figure()
        plt.imshow(road)
        plt.savefig(path_for_image_plots + 'render_road_and_borders_render_road_and_borders.png')
        plt.show()
        return road

    @staticmethod
    def plot_ego_and_ground_truth(gtxs, road, trajectory_gt, path_for_image_plots):
        light_grey = 0.8
        dark_grey = 0.4
        shape = (gtxs[0][4].shape[0], gtxs[0][4].shape[1], 3)

        # %matplotlib
        # inline

        figure = np.ones(shape)

        # draw ego in red
        ego_black = 1 - np.swapaxes(gtxs[0][4], 0, 1).numpy()
        ego = np.ones(shape)
        ego[:, :, 1] = ego_black
        ego[:, :, 2] = ego_black

        # draw ground truth in dark grey
        all_bb_black = 1 - np.swapaxes(gtxs[0][3], 0, 1).numpy()
        all_bb = np.ones(figure.shape)
        all_bb[all_bb_black == 0] = dark_grey

        # combine ego and ground truth bboxes
        all_bb[ego == 0] = 0

        # add trajectory in black
        all_bb[trajectory_gt == 1] = 0

        figure = all_bb

        plt.figure()
        plt.title('ego, bboxes, and trajectory ground truth')
        plt.imshow(figure)
        plt.savefig(path_for_image_plots + 'plot_ego_and_ground_truth_trajectory_in_black.png')

        plt.show()

        # add road in light grey
        figure[road == light_grey] = light_grey

        plt.figure()
        plt.title('ego, bboxes, and trajectory ground truth')
        plt.grid(None)
        plt.imshow(figure)
        plt.savefig(path_for_image_plots + 'plot_ego_and_ground_truth_road_in_light_grey_1.png')

        plt.show()

        # add road in light grey
        figure[road == light_grey] = light_grey

        plt.figure()
        plt.title('ego, bboxes, and trajectory ground truth')
        plt.grid(None)
        plt.imshow(figure)
        plt.savefig(path_for_image_plots + 'plot_ego_and_ground_truth_road_in_light_grey_2.png')

        plt.show()
        return road

    @staticmethod
    def plot_predicted_ego_with_bboxes(path_for_image_plots, road, gtxs, predxs_original, trajectory_pt):
        # %matplotlib
        # inline
        light_grey = 0.8
        dark_grey = 0.4
        shape = (gtxs[0][4].shape[0], gtxs[0][4].shape[1], 3)

        figure = np.ones(shape)

        # draw ego in red
        ego_black = 1 - np.swapaxes(predxs_original[0][4], 0, 1).numpy()
        ego = np.ones(shape)
        ego[:, :, 1] = ego_black
        ego[:, :, 2] = ego_black

        # draw ground truth in dark grey
        all_bb_black = 1 - np.swapaxes(predxs_original[0][3], 0, 1).numpy()
        all_bb = np.ones(figure.shape)
        all_bb[all_bb_black == 0] = dark_grey

        # combine ego and ground truth bboxes
        all_bb[ego == 0] = 0

        # add trajectory in black
        all_bb[trajectory_pt == 1] = 0

        figure = all_bb

        plt.figure()
        plt.title('ego, bboxes, and trajectory predicted original')
        plt.imshow(figure)
        plt.savefig(path_for_image_plots + 'plot_predicted_ego_with_bboxes_trajectory_in_black.png')

        plt.show()

        # add road in light grey
        figure[road == light_grey] = light_grey

        plt.figure()
        plt.title('ego, bboxes, and trajectory predicted original')
        plt.grid(None)
        plt.imshow(figure)
        plt.savefig(path_for_image_plots + 'plot_predicted_ego_with_bboxes_road_in_light_grey_3.png')
        plt.show()

    @staticmethod
    def show_road_with_dynamic_predicted_ego_and_bboxes(path_for_image_plots,road,gtxs,predxs_dynamic,trajectory_dt):
        # %matplotlib
        # inline
        light_grey = 0.8
        dark_grey = 0.4
        shape = (gtxs[0][4].shape[0], gtxs[0][4].shape[1], 3)

        figure = np.ones(shape)

        # draw ego in red
        ego_black = 1 - np.swapaxes(predxs_dynamic[0][4], 0, 1).numpy()
        ego = np.ones(shape)
        ego[:, :, 1] = ego_black
        ego[:, :, 2] = ego_black

        # draw ground truth in dark grey
        all_bb_black = 1 - np.swapaxes(predxs_dynamic[0][3], 0, 1).numpy()
        all_bb = np.ones(figure.shape)
        all_bb[all_bb_black == 0] = dark_grey

        # combine ego and ground truth bboxes
        all_bb[ego == 0] = 0

        # add trajectory in black
        all_bb[trajectory_dt == 1] = 0

        figure = all_bb

        plt.figure()
        plt.title('ego, bboxes, and trajectory predicted our solution')
        plt.imshow(figure)
        plt.savefig(path_for_image_plots + 'show_road_with_dynamic_predicted_ego_and_bboxes_trajectory_in_black.png')

        plt.show()

        # add road in light grey
        figure[road == light_grey] = light_grey

        plt.figure()
        plt.title('ego, bboxes, and trajectory predicted our solution')
        plt.grid(None)
        plt.imshow(figure)
        plt.savefig(path_for_image_plots + 'show_road_with_dynamic_predicted_ego_and_bboxes_road_in_light_grey.png')

        plt.show()
