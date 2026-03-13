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

        # 3. Interpolate (Resize) to match sim specification (720 samples)
        current_len = len(raw_ranges)
        target_len = self._NUM_SAMPLES  # 720

        if current_len != target_len:
            # 0 から (元のデータ数-1) までの範囲を、720分割するインデックスを作成
            new_indices = np.linspace(0, current_len - 1, target_len)
            # 最も近い整数のインデックスに丸める
            nearest_indices = np.round(new_indices).astype(int)
            # 元のデータから「一番近い点」をピックアップして 720個並べる
            self.__samples_new = raw_ranges[nearest_indices].astype(np.float32)
        else:
            self.__samples_new = raw_ranges.astype(np.float32)


        # 4. Restore Zeros
        # 補間が終わったので、遠すぎる値（20メートル以上など）は再び「0（無限遠・測定不能）」に戻してあげる

        # === 【追加修正：実機の設置角度（右90度回転）の補正】 ===
        # 720サンプル中、90度は 180サンプル(720 / 4)に相当。
        # 右に90度回っているデータを左に90度戻すため、インデックスを「180」ずらします。
        # np.roll(配列, シフト量) を使用。
        # ※実機の回転方向やデータの並び順により「180」か「-180」かは実機確認が必要ですが、
        #   右回転設置を戻す場合は通常「180」または「-180」で整合します。
        self.__samples_new = np.roll(self.__samples_new, -180)

    def __update(self):
        self.__samples = self.__samples_new

    def get_samples(self) -> np.ndarray[720, np.float32]:
        return self.__samples

    def get_samples_async(self) -> np.ndarray[720, np.float32]:
        return self.__samples_new