from shapely.ops import unary_union
from src.cores.base import StormsMap

def pod_score(pred_map: StormsMap, true_map: StormsMap) -> float:
    """
    Probability of Detection (POD) score for storms prediction. It is computed as the ratio of correctly predicted storms to the total area of true storms.
    High POD indicates that most of the true storms were detected.
    """
    multi_poly1 = unary_union([storm.contour for storm in pred_map.storms])
    multi_poly2 = unary_union([storm.contour for storm in true_map.storms])
    
    true_positive_area = multi_poly1.intersection(multi_poly2).area
    true_area = sum(storm.contour.area for storm in true_map.storms)

    return true_positive_area / true_area if true_area > 0 else 0.0

def far_score(pred_map: StormsMap, true_map: StormsMap) -> float:
    """
    False Alarm Ratio (FAR) for storms prediction. It is computed as the ratio of incorrectly predicted storms to the total area of predicted storms.
    Low FAR indicates that most of the predicted storms were correct.
    """
    multi_poly1 = unary_union([storm.contour for storm in pred_map.storms])
    multi_poly2 = unary_union([storm.contour for storm in true_map.storms])
    
    false_positive_area = multi_poly1.difference(multi_poly2).area
    pred_area = multi_poly1.area

    return false_positive_area / pred_area if pred_area > 0 else 0.0

def csi_score(pred_map: StormsMap, true_map: StormsMap) -> float:
    """
    Critical Success Index (CSI) for storms prediction. It is computed as the ratio of correctly predicted storms to the total area of predicted and true storms minus the area of correctly predicted storms.
    """
    multi_poly1 = unary_union([storm.contour for storm in pred_map.storms])
    multi_poly2 = unary_union([storm.contour for storm in true_map.storms])
    
    true_positive_area = multi_poly1.intersection(multi_poly2).area
    false_positive_area = multi_poly1.difference(multi_poly2).area
    false_negative_area = multi_poly2.difference(multi_poly1).area

    return true_positive_area / (true_positive_area + false_positive_area + false_negative_area) if (true_positive_area + false_positive_area + false_negative_area) > 0 else 0.0