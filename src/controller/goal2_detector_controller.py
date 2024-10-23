from model.DTOs.settings_model_for_goal2_detector import SettingsForGoal2Detector
from utils.class_implementations.collection_utils import CollectionUtils
from utils.class_implementations.files_utils import FileUtils
from model.Domain_models.goal2_detector_domain_model import GOAL2DetectorDomainModel
from utils.class_implementations.analyzer_utils import AnalyzerUtils
from utils.class_implementations.pkl_trajectory_analyzer_utils import PKLTrajectoryAnalyzer


class Goal2DetectorController:
    def __init__(self, settingsModel: SettingsForGoal2Detector):
        self.settingsModel = settingsModel

    def compute(self):
        settings = self.settingsModel
        utils = CollectionUtils()
        file_utils = FileUtils()
        scenes_list = utils.get_scenes_list(nuscenes=settings.nuscenes, val=settings.scene_for_eval_set)
        validation_samples = utils.get_validation_samples(scenes_list=scenes_list, nuscenes=settings.nuscenes)
        list_tokens = file_utils.save_tokens_to_json(validation_samples=validation_samples,
                                                     results_path=settings.result_path)
        model_object = utils.load_pkl_model(model_path=settings.model_path, mask_json=settings.mask_json,
                                            verbose=settings.verbose)
        dist_list = [2.0]
        detector_file = settings.path + list(settings.nuscenes_detectors.items())[0][1] + settings.file_json
        dt = utils.create_dt(detector_file,
                             'val',
                             model=list(settings.detector.items())[0][1],
                             d=settings.d,
                             r=settings.r,
                             t=settings.t,
                             verbose=settings.verbose)
        pkl_original = dt.calc_sample_crit_GOAL2(set_of_tokens=list_tokens,
                                                 conf_ths=[settings.threshold],
                                                 crit_list=[10.0],  # so nothing is included
                                                 obj_classes_list=settings.arrayOfObjectClasses,
                                                 # filter boxes based on class
                                                 verbose=settings.verbose)
        pkl_dynamic_results = dt.calc_sample_crit_GOAL2(set_of_tokens=list_tokens,
                                                        conf_ths=[settings.threshold],
                                                        crit_list=[settings.crit],
                                                        obj_classes_list=settings.arrayOfObjectClasses,
                                                        # filter boxes based on class
                                                        verbose=settings.verbose)
        original_better, dynamic_better, final_list = AnalyzerUtils.pkl_comparison_analyzer(
            pkl_dynamic_results=pkl_dynamic_results, pkl_original=pkl_original)
        trajectory_model = PKLTrajectoryAnalyzer.process_all_tokens(settingsModel=settings,
                                                                    token_list_in_use=dynamic_better, dt=dt)
        PKLTrajectoryAnalyzer.plot_trajectory(trajectory_model, path_for_image_plots=settings.path_for_image_plots)
        result = GOAL2DetectorDomainModel(pkl_mean=pkl_dynamic_results[0]['mean'],
                                          pkl_median=pkl_dynamic_results[0]['median'],
                                          original_mean=pkl_original[0]['mean'],
                                          original_median=pkl_original[0]['median'], original_better=original_better,
                                          dynamic_better=dynamic_better, final_list=final_list)

        return result

    def analyze_token_with_trajectory(self, single_token, dt, threshold=0.25):
        settings = self.settingsModel
        pklfile_dynamic, info_dynamic, all_pkls_dynamic, gtdist, preddist_dynamic, gtxs, predxs_dynamic, createdimages_dynamic = dt.calc_sample_crit_GOAL2_unique_param(
            listtoken=[single_token],
            conf_th=0.0,
            crit=10.0,
            obj_classes_list=settings.arrayOfObjectClasses,  # filter boxes based on class
            verbose=settings.verbose)
        (pklfile_original, info_original, all_pkls_original, gtdist, preddist_original, gtxs, predxs_original,
         createdimages_original) = dt.calc_sample_crit_GOAL2_unique_param(
            listtoken=[single_token],
            conf_th=threshold,
            crit=10.00,
            obj_classes_list=settings.arrayOfObjectClasses,
            # filter boxes based on class
            verbose=settings.verbose)
        trajectory_gt, trajectory_pt = PKLTrajectoryAnalyzer.visualize_trajectory_and_detections(
            createdimages_dynamic=createdimages_dynamic,
            info_original=info_original,
            createdimages_original=createdimages_original,
            info_dynamic=info_dynamic,
            path_for_image_plots=settings.path_for_image_plots)
        road = PKLTrajectoryAnalyzer.render_road_and_borders(path_for_image_plots=settings.path_for_image_plots,
                                                             gtxs=gtxs)
        road = PKLTrajectoryAnalyzer.plot_ego_and_ground_truth(path_for_image_plots=settings.path_for_image_plots,
                                                               gtxs=gtxs,
                                                               trajectory_gt=trajectory_gt, road=road)
        PKLTrajectoryAnalyzer.plot_predicted_ego_with_bboxes(road=road, gtxs=gtxs,
                                                             path_for_image_plots=settings.path_for_image_plots,
                                                             predxs_original=predxs_original,
                                                             trajectory_pt=trajectory_pt)
        PKLTrajectoryAnalyzer.show_road_with_dynamic_predicted_ego_and_bboxes(path_for_image_plots=settings.path_for_image_plots,road=road,gtxs=gtxs,predxs_dynamic=predxs_dynamic,trajectory_dt=trajectory_gt)
