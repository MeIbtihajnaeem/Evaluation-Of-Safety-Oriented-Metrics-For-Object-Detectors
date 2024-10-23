class GOAL2DetectorDomainModel:
    def __init__(self, original_mean, original_median, pkl_mean, pkl_median,

                 original_better, dynamic_better, final_list,dt):
        self.original_mean = original_mean
        self.original_median = original_median
        self.pkl_mean = pkl_mean
        self.pkl_median = pkl_median
        self.original_better = original_better
        self.dynamic_better = dynamic_better
        self.final_list = final_list
        self.dt = dt

    def get_original_better(self):
        return self.original_better

    def get_dynamic_better(self):
        return self.dynamic_better
    def get_dt(self):
        return self.dt

    def get_final_list(self):
        return self.final_list

    def get_original_mean(self):
        return self.original_mean

    def get_original_median(self):
        return self.original_median

    def get_pkl_mean(self):
        return self.pkl_mean

    def get_pkl_median(self):
        return self.pkl_median

    def print_comparison(self):
        print("pkl mean with original approach {}, versus GOAL2 approach {}".format(
            self.original_mean, self.pkl_mean))

        print("pkl median with original approach {}, versus GOAL2 approach {}".format(
            self.original_median, self.pkl_median))
