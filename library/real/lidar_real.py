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

        # 初期化時のサイズ指定
        # 親クラスで規定されたサイズ(720)のゼロ配列で初期化します。
        self.__samples = np.zeros(self._NUM_SAMPLES, dtype=np.float32)
        self.__samples_new = np.zeros(self._NUM_SAMPLES, dtype=np.float32)

    def __scan_callback(self, data):
        # 1. Convert to numpy array & meters to cm
        raw_ranges = np.array(data.ranges) * 100

        # 2. Flip to correct for CW spin - matches with sim
        # X4からX2への移行により左右反転が不要になったためコメントアウト
        # raw_ranges = np.flip(raw_ranges)

        # === 【修正：オバケ（補間ノイズ）対策】 ===
        # 「0(なし)」や「inf」と「壁」の間を補間すると、中間の値（近距離の壁）が
        # 生まれてしまうため、一時的に「遠くの値」に置き換えてから計算します。

        MASK_VALUE = 3000.0  # 30メートル（十分に大きな値）
        
        # inf(無限大)、nan(非数)、0(測定不能) をすべて 3000.0 に置換
        raw_ranges[np.isinf(raw_ranges)] = MASK_VALUE
        raw_ranges[np.isnan(raw_ranges)] = MASK_VALUE
        raw_ranges[raw_ranges == 0] = MASK_VALUE

        # 3. Interpolate (Resize) to match sim specification (720 samples)
        current_len = len(raw_ranges)
        target_len = self._NUM_SAMPLES  # 720

        if current_len != target_len:
            # 0から1までの区間を、現在の個数と目標の個数で等分割するインデックスを作成
            old_indices = np.linspace(0, 1, current_len)
            new_indices = np.linspace(0, 1, target_len)
            
            # 線形補間を実行（この時点では、壁の隣は3000に向かってなだらかに増える）
            self.__samples_new = np.interp(new_indices, old_indices, raw_ranges).astype(np.float32)
        else:
            self.__samples_new = raw_ranges.astype(np.float32)

        # 4. Restore Zeros
        # 補間が終わったので、遠すぎる値（20メートル以上など）は再び「0（無限遠・測定不能）」に戻してあげる
        self.__samples_new[self.__samples_new > 2000.0] = 0.0

    def __update(self):
        self.__samples = self.__samples_new

    def get_samples(self) -> np.ndarray[720, np.float32]:
        return self.__samples

    def get_samples_async(self) -> np.ndarray[720, np.float32]:
        return self.__samples_new