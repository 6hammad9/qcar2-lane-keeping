#!/usr/bin/env python3

import os
import subprocess
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory


class SoundNode(Node):
    def __init__(self):
        super().__init__("qcar2_sound_node")

        self.sound_device = "demixer"

        self.sound_dir = os.path.join(
            get_package_share_directory("qcar_science_night_pkg"),
            "config",
        )

        self.sound_files = {
            "obstacle": "obstacle.wav",
            "pickup": "pickup.wav",
            "resume": "resume.wav",
            "mission_complete": "mission_complete.wav",
            "delivery": "delivery.wav",
            "complete": "complete.wav",
        }

        self.is_playing = False
        self.lock = threading.Lock()

        self.subscription = self.create_subscription(
            String,
            "/qcar2/sound_event",
            self.handle_sound,
            10,
        )

        self.get_logger().info(f"QCar2 WAV Sound Node Ready. Sound dir: {self.sound_dir}")

    def play_sound_worker(self, sound_path):
        try:
            subprocess.run(
                [
                    "aplay",
                    "-q",
                    "-D",
                    self.sound_device,
                    sound_path,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"aplay failed: {e}")

        except Exception as e:
            self.get_logger().error(f"Failed to play sound: {e}")

        finally:
            with self.lock:
                self.is_playing = False

    def play_sound(self, sound_path):
        if not os.path.exists(sound_path):
            self.get_logger().error(f"Sound file not found: {sound_path}")
            return

        with self.lock:
            if self.is_playing:
                self.get_logger().warn("Sound already playing, skipping new event")
                return

            self.is_playing = True

        thread = threading.Thread(
            target=self.play_sound_worker,
            args=(sound_path,),
            daemon=True,
        )
        thread.start()

    def handle_sound(self, msg):
        event = msg.data.strip().lower()

        if event not in self.sound_files:
            self.get_logger().warn(f"Unknown sound event: {msg.data}")
            return

        sound_path = os.path.join(self.sound_dir, self.sound_files[event])

        self.get_logger().info(f"Playing sound event: {event}")
        self.play_sound(sound_path)


def main(args=None):
    rclpy.init(args=args)
    node = SoundNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()