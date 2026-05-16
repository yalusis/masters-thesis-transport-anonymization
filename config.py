"""
Конфігурація параметрів анонімізації та структури датасету UAH-DriveSet
"""

# Параметри анонімізації
K_ANONYMITY = 3        # мінімальний розмір класу еквівалентності
L_DIVERSITY = 2        # мінімальна кількість різних значень SA у класі
GEOHASH_PRECISION = 5  # точність геохешу
TIME_BUCKET_SECONDS = 300

ONTOLOGY_IRI = "http://uah.edu.ua/ontologies/transport-anonymization.owl"
TRANSPORT_NS_URI = "http://uah.edu.ua/ontologies/transport#"

#Стовпці SEMANTIC_ONLINE.txt
ONLINE_COLUMNS = [
    "timestamp_sec",           # 1. Час з початку маршруту (секунди)
    "gps_latitude",            # 2. GPS широта
    "gps_longitude",           # 3. GPS довгота
    "score_total_window",      # 4. Загальна оцінка
    "score_acc_window",        # 5. Оцінка прискорень
    "score_brake_window",      # 6. Оцінка гальмувань
    "score_turn_window",       # 7. Оцінка поворотів
    "score_weave_window",      # 8. Оцінка плетіння смуг
    "score_drift_window",      # 9. Оцінка дрейфу смуги
    "score_overspeed_window",  # 10. Оцінка перевищення швидкості
    "score_carfollow_window",  # 11. Оцінка дистанції до авто
    "ratio_normal_window",     # 12. Частка нормальної їзди
    "ratio_drowsy_window",     # 13. Частка сонливої їзди
    "ratio_aggressive_window", # 14. Частка агресивної їзди
    "ratio_distracted_window", # 15. Відволікання
    "score_total",             # 16. Загальна оцінка
    "score_acc",               # 17. Оцінка прискорень
    "score_brake",             # 18. Оцінка гальмувань
    "score_turn",              # 19. Оцінка поворотів
    "score_weave",             # 20. Оцінка плетіння смуг
    "score_drift",             # 21. Оцінка дрейфу
    "score_overspeed",         # 22. Оцінка швидкості
    "score_carfollow",         # 23. Оцінка дистанції
    "ratio_normal",            # 24. Частка норм. їзди
    "ratio_drowsy",            # 25. Частка сонливої
    "ratio_aggressive",        # 26. Частка агресивної
    "ratio_distracted",        # 27. Відволікання
]

# Поля SEMANTIC_FINAL.txt
FINAL_FIELDS = {
    0: "hour_start", 1: "minute_start", 2: "second_start",
    3: "avg_speed_kmh", 4: "max_speed_kmh", 5: "lanex_score",
    6: "driving_time_min",
    7: "hour_end", 8: "minute_end", 9: "second_end",
    10: "trip_distance_km",
    11: "score_long_dist", 12: "score_tran_dist", 13: "score_speed_dist",
    14: "score_global",
    15: "alerts_long", 16: "alerts_late", 17: "alerts_lanex",
    18: "num_stops", 19: "speed_variability", 20: "acceleration_noise",
    21: "kinetic_energy", 22: "driving_time_sec", 23: "num_curves",
    24: "power_exerted",
    25: "acc_events", 26: "brake_events", 27: "turn_events",
    28: "long_distraction_global", 29: "trans_distraction_global",
    30: "mean_long_dist", 31: "std_long_dist",
    32: "avg_trans_dist", 33: "std_trans_dist",
    34: "lacc", 35: "macc", 36: "hacc",
    37: "lbra", 38: "mbra", 39: "hbra",
    40: "ltur", 41: "mtur", 42: "htur",
    43: "score_total_final",
    44: "score_acc_final", 45: "score_brake_final",
    46: "score_turn_final", 47: "score_lane_weave_final",
    48: "score_lane_drift_final", 49: "score_overspeed_final",
    50: "score_carfollow_final",
    51: "ratio_normal_final", 52: "ratio_drowsy_final",
    53: "ratio_aggressive_final",
}

# Порогові значення для категоризації швидкості
SPEED_THRESHOLDS = [0, 30, 90, float('inf')]
SPEED_LABELS = ['low', 'medium', 'high']

# Точність геохешу, приблизний радіус комірки
GEOHASH_PRECISION_KM = {
    1: 2500.0,
    2: 630.0,
    3: 78.0,
    4: 20.0,
    5: 2.4,
    6: 0.61,
    7: 0.076,
    8: 0.019,
}
