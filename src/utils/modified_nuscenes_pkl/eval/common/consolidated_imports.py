# importing data_classes custom data_classes

# importing loaders from modified nuscenes
# importing loaders from nuscenes library
from nuscenes.eval.common.loaders import add_center_dist

# importing from utils modified nuscenes
# importing from utils nuscenes library
from nuscenes.eval.common.utils import boxes_to_sensor, center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, \
    cummean

__all__ = ['import_all', 'import_loaders', 'import_utils']


def import_all():
    # global EvalBox, EvalBoxType, EvalBoxes,
    # MetricData, load_prediction, load_gt,
    # filter_eval_boxes, add_center_dist, DetectionBox,
    # boxes_to_sensor_debug, boxes_to_sensor_crit, boxes_to_sensor,
    # center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
    return (
        add_center_dist, boxes_to_sensor, center_distance, scale_iou,
        yaw_diff, velocity_l2, attr_acc, cummean)


def import_loaders():
    # global load_prediction, load_gt, filter_eval_boxes, add_center_dist
    return add_center_dist


def import_utils():
    # global DetectionBox, boxes_to_sensor_debug,
    # boxes_to_sensor_crit, boxes_to_sensor, center_distance,
    # scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
    return (boxes_to_sensor,
            center_distance, scale_iou, yaw_diff,
            velocity_l2, attr_acc, cummean)
