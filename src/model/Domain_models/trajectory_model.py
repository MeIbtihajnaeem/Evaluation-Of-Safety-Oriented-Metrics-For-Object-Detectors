class TrajectoryModel:
    def __init__(self, font, trajectory_gt, trajectory_pt, trajectory_gt_dt, final_gt_dt,
                 ground_truths_dt, trajectory_dt, ground_truths, final_pt,
                 dynamic_detections, final_gt, final_dt, original_detections):
        self.font = font
        self.trajectory_gt = trajectory_gt
        self.trajectory_pt = trajectory_pt
        self.trajectory_gt_dt = trajectory_gt_dt
        self.final_gt_dt = final_gt_dt
        self.ground_truths_dt = ground_truths_dt
        self.trajectory_dt = trajectory_dt
        self.ground_truths = ground_truths
        self.final_pt = final_pt
        self.dynamic_detections = dynamic_detections
        self.final_gt = final_gt
        self.final_dt = final_dt
        self.original_detections = original_detections

    # Getters
    def get_font(self):
        return self.font

    def get_trajectory_gt(self):
        return self.trajectory_gt

    def get_trajectory_pt(self):
        return self.trajectory_pt

    def get_trajectory_gt_dt(self):
        return self.trajectory_gt_dt

    def get_final_gt_dt(self):
        return self.final_gt_dt

    def get_ground_truths_dt(self):
        return self.ground_truths_dt

    def get_trajectory_dt(self):
        return self.trajectory_dt

    def get_ground_truths(self):
        return self.ground_truths

    def get_final_pt(self):
        return self.final_pt

    def get_dynamic_detections(self):
        return self.dynamic_detections

    def get_final_gt(self):
        return self.final_gt

    def get_final_dt(self):
        return self.final_dt

    def get_original_detections(self):
        return self.original_detections

    # Method to display all information
    def display_info(self):
        print(f"Font: {self.font}")
        print(f"Trajectory GT: {self.trajectory_gt}")
        print(f"Trajectory PT: {self.trajectory_pt}")
        print(f"Trajectory GT_DT: {self.trajectory_gt_dt}")
        print(f"Final GT_DT: {self.final_gt_dt}")
        print(f"Ground Truths DT: {self.ground_truths_dt}")
        print(f"Trajectory DT: {self.trajectory_dt}")
        print(f"Ground Truths: {self.ground_truths}")
        print(f"Final PT: {self.final_pt}")
        print(f"Dynamic Detections: {self.dynamic_detections}")
        print(f"Final GT: {self.final_gt}")
        print(f"Final DT: {self.final_dt}")
        print(f"Original Detections: {self.original_detections}")

