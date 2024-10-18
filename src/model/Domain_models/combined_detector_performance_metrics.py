from .detector_performance_metrics import DetectorPerformanceMetrics


class CombinedDetectorPerformanceMetrics:
    def __init__(self, pkl_metrics: DetectorPerformanceMetrics, original_metrics: DetectorPerformanceMetrics = None):
        self._pkl_metrics = pkl_metrics
        self._original_metrics = original_metrics

    def get_pkl_metrics(self):
        return self._pkl_metrics

    def get_original_metrics(self):
        return self._original_metrics

    def print_all_metrics(self):
        print("PKL Metrics:")
        self._print_metrics(self._pkl_metrics)

        if self._original_metrics:
            print("\nOriginal Metrics:")
            self._print_metrics(self._original_metrics)
        else:
            print("\nOriginal Metrics: None")

    @staticmethod
    def _print_metrics(metrics: DetectorPerformanceMetrics):
        metrics.print_metrics()

    def calculate_differences(self):
        # Check if original_metrics is None and handle it accordingly
        if self._original_metrics is None:
            return {
                "goal": self._pkl_metrics.get_goal().name,  # No comparison, assume they are different
                "saved_pkl_mean": self._pkl_metrics.get_saved_pkl_mean(),
                "lowest_mean": self._pkl_metrics.get_lowest_mean(),
                "saved_pkl_median": self._pkl_metrics.get_saved_pkl_median(),
                "lowest_median": self._pkl_metrics.get_lowest_median(),
                "max_pkl": self._pkl_metrics.get_max_pkl(),
                "lowest_max": self._pkl_metrics.get_lowest_max(),
                "mean_ap": self._pkl_metrics.get_mean_ap(),
                "mean_ap_max": self._pkl_metrics.get_mean_ap_max(),
                "mean_ap_max_criteria": self._pkl_metrics.get_mean_ap_max_criteria()
            }
        print("----------")
        print(self._pkl_metrics.get_saved_pkl_mean())
        print(self._original_metrics.get_saved_pkl_mean())
        differences = {
            "goal": self._pkl_metrics.get_goal().name,
            "saved_pkl_mean": self._pkl_metrics.get_saved_pkl_mean(),
            "saved_pkl_mean_original": self._original_metrics.get_saved_pkl_mean(),
            "lowest_mean": self._pkl_metrics.get_lowest_mean() - self._original_metrics.get_lowest_mean(),
            "saved_pkl_median": self._pkl_metrics.get_saved_pkl_median() ,
            "saved_pkl_median_original":self._original_metrics.get_saved_pkl_median(),
            "lowest_median": self._pkl_metrics.get_lowest_median() - self._original_metrics.get_lowest_median(),
            "max_pkl": [a - b for a, b in zip(self._pkl_metrics.get_max_pkl(), self._original_metrics.get_max_pkl())],
            "lowest_max": self._pkl_metrics.get_lowest_max() - self._original_metrics.get_lowest_max(),
            "mean_ap": self._pkl_metrics.get_mean_ap() - self._original_metrics.get_mean_ap(),
            "mean_ap_max": self._pkl_metrics.get_mean_ap_max() - self._original_metrics.get_mean_ap_max(),
            "mean_ap_max_criteria": self._pkl_metrics.get_mean_ap_max_criteria()
        }

        return differences
