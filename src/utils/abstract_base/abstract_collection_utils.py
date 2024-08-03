from abc import ABC, abstractmethod
from src.enumerations import GOAL


class AbstractCollectionUtils(ABC):

    @abstractmethod
    def compute_crit_pkl(self, dt, criticalities, goal: GOAL, list_token=None,
                         conf_th_list=None, dist_list=None, object_classes='car',
                         verbose=False, model_loaded=False, model_object=None):
        pass

    @abstractmethod
    def get_scenes_list(self, nuscenes, val):
        pass

    @abstractmethod
    def get_validation_samples(self, scenes_list, nuscenes):
        pass

    @abstractmethod
    def get_list_token(self, validation_samples):
        pass

    @abstractmethod
    def create_dt(self, detector_file=None, val=None, model=None, d=1, r=1, t=1, verbose=False,
                  nuscenes=None, conf_value=None, n_workers=None, bsz=None, gpu_id=None, output_dir=None, crit=None):
        pass

    @abstractmethod
    def load_pkl_model(self, model_path, mask_json, verbose):
        pass

    @abstractmethod
    def plot_normal(self, data, confidence_threshold):
        pass

    @abstractmethod
    def plot_special(self, data, confidence_threshold,save_in,d,r,t):
        pass

    @abstractmethod
    def _prc(self,prec_rec_conf, target_confidence):
        pass

    @abstractmethod
    def plot_normal_and_special(self, special_05, special_1, special_2, special_4, normal_05, normal_1, normal_2,
                                normal_4, save_in, confidence_threshold, detector, d, r, t):
        pass
