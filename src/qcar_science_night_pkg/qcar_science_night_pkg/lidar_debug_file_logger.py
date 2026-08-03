import csv
import os
import time
from pathlib import Path


class LidarDebugFileLogger:
    def __init__(self, file_path):
        self.file_path = Path(file_path).expanduser()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        self.fieldnames = [
            "time",

            # Node decision info
            "drive_state",
            "lane_active",
            "lane_offset",
            "target_v_before",
            "target_v_after",
            "target_steer_before",
            "target_steer_after",
            "safety_stop",
            "stop_reason",

            # Final obstacle status
            "obstacle_ahead",
            "emergency",
            "left_clear",
            "right_clear",

            # Raw analyzer values
            "obstacle_ahead_raw",
            "emergency_raw",
            "left_clear_raw",
            "right_clear_raw",

            # Distances
            "front_min",
            "left_min",
            "right_min",
            "emergency_min",

            # Counts
            "front_count",
            "left_count",
            "right_count",
            "emergency_count",

            # Debounce streaks
            "front_clear_streak",
            "emergency_clear_streak",
            "left_clear_streak",
            "right_clear_streak",
        ]

        file_exists = self.file_path.exists()

        self.file = open(self.file_path, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)

        if not file_exists or os.path.getsize(self.file_path) == 0:
            self.writer.writeheader()
            self.file.flush()

    def write(self, **kwargs):
        row = {name: "" for name in self.fieldnames}
        row["time"] = time.time()

        for key, value in kwargs.items():
            if key in row:
                row[key] = value

        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        try:
            self.file.flush()
            self.file.close()
        except Exception:
            pass