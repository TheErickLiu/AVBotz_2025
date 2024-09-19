import argparse
import math
import serial
from typing import List, Tuple

SENSOR_GAME_ROTATION_VECTOR = 15
LEN_GAME_ROTATION_VECTOR = 6


# Convert quaternion (qx, qy, qz, qw) to Euler angles (yaw, pitch, roll) in degrees
# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
class Quaternion:
    def __init__(self, qlist: List[float]):
        """Init with [w, x, y, z]"""
        self.x = qlist[0]
        self.y = qlist[1]
        self.z = qlist[2]
        self.w = qlist[3]

    def to_euler_angles(self) -> Tuple[float, float, float]:
        """Convert quaternion to Euler angles (yaw, pitch, roll) in degrees"""

        # Roll: x-axis rotation
        sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1.0 - 2.0 * (self.x**2 + self.y**2)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch: y-axis rotation
        sinp = 2.0 * (self.w * self.y - self.z * self.x)
        sinp = 1.0 if sinp > 1.0 else sinp
        sinp = -1.0 if sinp < -1.0 else sinp
        pitch = math.asin(sinp)

        # Yaw: z-axis rotation
        siny_cosp = 2.0 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1.0 - 2.0 * (self.y**2 + self.z**2)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Question2 to read PNI NaviGuider with UART."
    )
    # Linux uses ttySx as the serial port device name.
    parser.add_argument(
        "--dev",
        type=str,
        required=True,
        help="Serial device name of the PNI device (e.g. /dev/ttyS0).",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=50,
        help="Sample rate for Quaternions sensors (0-400Hz)",
    )

    return parser.parse_args()


def get_serial(dev: str, rate: int) -> serial.Serial:
    # Open the serial port after connecting the USB-serial cable (Figure 2-1)
    # Serial configurations from Table 3-1 Communication Format
    # https://pyserial.readthedocs.io/en/latest/
    ser = serial.Serial(
        port=dev,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
    )

    # Configure and start the
    # Table 4-1 Summary of Simple Serial Character Commands
    command = f"s {SENSOR_GAME_ROTATION_VECTOR},{rate}\r"
    ser.write(command.encode())

    # Sensor data display is ON by default. Otherwise, it needs
    # to be turned on with the D1 command.
    # Disable verbose mode to get just data.
    ser.write(b"V0\r")

    return ser


def read_data(ser: serial.Serial):
    try:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                # Comma separated data: UART Output Format
                # Timestamp,SensorID[,Value][,Value]...[,Value]
                data = line.split(",")
                if (len(data) == LEN_GAME_ROTATION_VECTOR) and (
                    int(data[1]) == SENSOR_GAME_ROTATION_VECTOR
                ):
                    q = Quaternion(data[2:6])
                    roll, pitch, yaw = q.to_euler_angles()
                    print(f"Yaw: {yaw}, Pitch: {pitch}, Roll: {roll}")

    except KeyboardInterrupt:
        ser.close()

args = parse_args()
ser = get_serial(args.dev, args.rate)
read_data(ser)
