class DetectorPerformanceMetrics:
    def __init__(self, goal, saved_pkl_mean, lowest_mean, saved_pkl_median, lowest_median, max_pkl, lowest_max,
                 mean_ap, mean_ap_max, mean_ap_max_criteria):
        self._goal = goal
        self._saved_pkl_mean = saved_pkl_mean
        self._lowest_mean = lowest_mean
        self._saved_pkl_median = saved_pkl_median
        self._lowest_median = lowest_median
        self._max_pkl = max_pkl
        self._lowest_max = lowest_max
        self._mean_ap = mean_ap
        self._mean_ap_max = mean_ap_max
        self._mean_ap_max_criteria = mean_ap_max_criteria

    # Getters for all attributes
    def get_goal(self):
        return self._goal

    def get_saved_pkl_mean(self):
        return self._saved_pkl_mean

    def get_lowest_mean(self):
        return self._lowest_mean

    def get_saved_pkl_median(self):
        return self._saved_pkl_median

    def get_lowest_median(self):
        return self._lowest_median

    def get_max_pkl(self):
        return self._max_pkl

    def get_lowest_max(self):
        return self._lowest_max

    def get_mean_ap(self):
        return self._mean_ap

    def get_mean_ap_max(self):
        return self._mean_ap_max

    def get_mean_ap_max_criteria(self):
        return self._mean_ap_max_criteria

    # Method to print all the data
    def print_metrics(self):
        print(f"Goal: {self._goal}")
        print(f"Saved PKL Mean: {self._saved_pkl_mean}")
        print(f"Lowest Mean: {self._lowest_mean}")
        print(f"Saved PKL Median: {self._saved_pkl_median}")
        print(f"Lowest Median: {self._lowest_median}")
        print(f"Max PKL: {self._max_pkl}")
        print(f"Lowest Max: {self._lowest_max}")
        print(f"Mean AP: {self._mean_ap}")
        print(f"Mean AP Max: {self._mean_ap_max}")
        print(f"Mean AP Max Criteria (D, R, T): {self._mean_ap_max_criteria}")
