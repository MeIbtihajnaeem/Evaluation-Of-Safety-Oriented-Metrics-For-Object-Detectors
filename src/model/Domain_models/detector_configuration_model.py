class DetectorConfigurationModel:
    def __init__(self, d, r, t, confidence_threshold, criticality_threshold):
        self.d = d
        self.r = r
        self.t = t
        self.confidence_threshold = confidence_threshold
        self.criticality_threshold = criticality_threshold

