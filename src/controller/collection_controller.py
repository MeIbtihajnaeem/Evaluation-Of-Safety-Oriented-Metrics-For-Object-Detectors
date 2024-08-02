from ..model.settings_model import SettingsModel
import itertools
import json
from nuscenes import NuScenes

# custom imports
import nuscenes.eval.detection.config as config


class CollectionController:

    def __init__(self, settingsModel: SettingsModel):
        self.settingsModel = settingsModel
        self.__result_path = settingsModel.notebook_home + "'pkl/results/GOAL2/retry_allobjects/"
        self.__drt = list(itertools.product(*[settingsModel.maxD, settingsModel.maxR, settingsModel.maxT]))
        self.__eval = settingsModel.scene_for_eval_set
        self.__val = settingsModel.scene_for_eval_set
        self.__nuscenes = NuScenes('v1.0-trainval', dataroot=settingsModel.data_root)
        self.__conf_value = config.config_factory("detection_cvpr_2019")
        self.__scenes_list = self._get_scenes_list(self.__nuscenes, self.__val)
        self.__validation_samples = self._get_validation_samples(self.__scenes_list, self.__nuscenes)
        self.__list_token = self._get_list_token(self.__validation_samples)
        with open(self.__result_path + "/token_list.json", "w") as outfile:
            json.dump(self.__validation_samples, outfile)

    def _method_for_goal1(self, drt):
        drt = self.__drt

    @staticmethod
    def _get_scenes_list(nuscenes, val):
        scenes_list = []
        counter = 0
        for i in nuscenes.scene:
            name = i['name']
            if name in val:
                counter = counter + 1
                scenes_list.append(i)
        return scenes_list

    @staticmethod
    def _get_validation_samples(__scenes_list, __nuscenes):
        validation_samples = {}
        for i in __scenes_list:
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
                sample = __nuscenes.get('sample', current_sample_token)
                current_sample_token = sample['next']
                sample_token_list.append(current_sample_token)
            if sample_token_list[len(sample_token_list) - 1] != last_sample_token:
                print("error")
                break
            validation_samples.update({scene_name: sample_token_list})
        return validation_samples

    @staticmethod
    def _get_list_token(validation_samples):
        list_token = []
        for i in validation_samples.keys():
            list_token.extend(validation_samples[i])
        return list_token
