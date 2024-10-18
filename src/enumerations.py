from enum import Enum
from model.Domain_models.detector_configuration_model import DetectorConfigurationModel


class GOAL(Enum):
    goal1 = "GOAL1"
    goal2 = "GOAL2"


class OBJECT_CLASSES(Enum):
    car = "car"
    truck = "truck"
    bus = "bus"
    trailer = "trailer"
    construction_vehicle = "construction_vehicle"
    pedestrian = "pedestrian"
    motorcycle = "motorcycle"
    bicycle = "bicycle"
    traffic_cone = "traffic_cone"
    barrier = "barrier"


class DETECTOR(Enum):
    SECFPN = "SECFPN"
    FCOS3D = "FCOS3D"
    PGD = "PGD"
    POINTP = "POINTP"
    REG = "REG"
    SSN = "SSN"

    def get_configuration(self):
        if self == DETECTOR.FCOS3D:
            return DetectorConfigurationModel(d=50, r=10, t=24, confidence_threshold=0.25, criticality_threshold=0.30)
        elif self == DETECTOR.PGD:
            return DetectorConfigurationModel(d=50, r=50, t=24, confidence_threshold=0.05, criticality_threshold=0.2)
        elif self == DETECTOR.POINTP:
            return DetectorConfigurationModel(d=5, r=5, t=4, confidence_threshold=0.55, criticality_threshold=0.65)
        elif self == DETECTOR.REG:
            return DetectorConfigurationModel(d=5, r=5, t=4, confidence_threshold=0.5, criticality_threshold=0.65)
        elif self == DETECTOR.SECFPN:
            return DetectorConfigurationModel(d=10, r=5, t=8, confidence_threshold=0.4, criticality_threshold=0.65)
        elif self == DETECTOR.SSN:
            return DetectorConfigurationModel(d=45, r=45, t=40, confidence_threshold=0.25, criticality_threshold=0.15)
        else:
            raise ValueError("Unknown detector type")
