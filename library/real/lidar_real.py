"""
Copyright MIT
MIT License

BWSI Autonomous RACECAR Course
Racecar Neo LTS

File Name: lidar_real.py
File Description: Contains the Lidar module of the racecar_core library
"""

from lidar import Lidar

# General
import numpy as np

# ROS2
import rclpy as ros2
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class LidarReal(Lidar):
    # The ROS topic from which we get Lidar data
    __SCAN_TOPIC = "/scan"

    def __init__(self):
        # ROS node
        self.node = ros2.create_node("scan_sub")

        # subscribe to the scan topic, which will call
        # __scan_callback every time the lidar sends data
        self.__scan_sub = self.node.create_subscription(
            LaserScan, self.__SCAN_TOPIC, self.__scan_callback, qos_profile_sensor_data
        )

        self.__samples = np.zeros(self._NUM_SAMPLES, dtype=np.float32)
        self.__samples_new = np.zeros(self._NUM_SAMPLES, dtype=np.float32)

    def __scan_callback(self, data):
        # 1. Convert to numpy array & meters to cm
        raw_ranges = np.array(data.ranges) * 100

        # 2. Interpolate (Resize) to match sim specification (720 samples)
        current_len = len(raw_ranges)
        target_len = self._NUM_SAMPLES 

        if current_len != target_len:
            new_indices = np.linspace(0, current_len - 1, target_len)
            nearest_indices = np.round(new_indices).astype(int)
            self.__samples_new = raw_ranges[nearest_indices].astype(np.float32)
        else:
            self.__samples_new = raw_ranges.astype(np.float32)


        # 3. Restore Zeros
        self.__samples_new = np.roll(self.__samples_new, -180)

    def __update(self):
        self.__samples = self.__samples_new

    def get_samples(self) -> np.ndarray[720, np.float32]:
        return self.__samples

    def get_samples_async(self) -> np.ndarray[720, np.float32]:
        return self.__samples_new