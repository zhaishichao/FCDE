from .ds_smote import DSSMOTE
# 四个约束，两个目标 原始版本的GSSMOTE
# max_g1 = max(avg_distance - ind.distance_minority_min for ind in individuals)  # distance_minority_min > 0
# max_g2 = max(0 - ind.fitness.values[0] for ind in individuals)  # ind.fitness.values[0] (第一个目标) > 0
# max_g3 = max(ind.distance_minority_center for ind in individuals)  # distance_minority_center - ave_max_distance > 0
# max_g4 = max((ind.cosine_angle - 90) for ind in individuals)  # cosine_angle < 90