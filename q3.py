import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class Controller:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp # Proportional gain
        self.ki = ki # Integral gain
        self.kd = kd # Derivative gain
        self.totalError = 0
        self.prevError = 0
        self.verbose = False

    def output(self, currentTilt: float, setPoint: float, timeStep: float) -> float:
        """Calculate PID control torque"""

        error = setPoint - currentTilt
        self.totalError += error * timeStep
        dError = (error - self.prevError) / timeStep
        if self.verbose:
            print("totalError =", self.totalError, "dError =", dError)

        torque = (self.kp * error) + (self.ki * self.totalError) + (self.kd * dError)
        if self.verbose:
            print("torque =", torque)

        self.prevError = error
        return torque


def random_noise(
    diste: float, timestep: int, magnitude: int = 1000, freq: int = 30
) -> Tuple[float, float]:
    """Calculate a random noise with a magnitude (meaning it can be negative, too) 
    between 0N to 1000N from a distance between 0 and diste meters on every 30th timestep
    """
    if timestep % freq == 0:
        magnitude = np.random.uniform(-magnitude, magnitude)
        distance = np.random.uniform(0, diste)
        return magnitude, distance
    return 0, 0


def plot_angles_over_time(times: list, angles: list, filename: str = "tilt.png"):
    """Plot simulation results and save figure"""
    plt.figure(figsize=(10, 5))
    plt.plot(times, angles, label="Tilt Angle (Degrees)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Tilt Angle (degrees)")
    plt.title("Robot Tilt Angle over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

class Simulation:
    def __init__(
        self,
        controller: Controller, # PID controller
        robotMass: float,  # Example mass in kg
        g: float,  # Gravity in m/s^2
        I: float,  # Rotational inertia in kg*m^2
        dist: float,  # Distance from rotational axis to motor in meters
        diste: float,  # Maximum distance from rotational axis to noise force in meters
        timeStep: float,  # Time step for simulation in seconds
        currentPos: float,  # Initial tilt angle in degrees
    ):
        self.controller = controller
        self.robotMass = robotMass
        self.g = g
        self.I = I
        self.dist = dist
        self.diste = diste
        self.timeStep = timeStep
        self.currentPos = currentPos
        self.angles = [self.currentPos]
        self.time = [0]
        self.n = 0

    def output(self, setpoint: float) -> float:
        """Torque being applied by PID Controller"""
        return self.controller.output(self.currentPos, setpoint, self.timeStep)

    def noise(self) -> Tuple[float, float]:
        """Calculate random noise"""
        return random_noise(self.diste, self.n)
    

    def simulate(self, setpoint: float, totalTime: float):
        """Simulate with given setpoint and simulation total seconds"""
        numSteps = int(totalTime / self.timeStep)

        for _ in range(numSteps):
            torque = self.output(setpoint)
            noiseForce, noiseDistance = self.noise()

            noiseTorque = (
                noiseForce * noiseDistance * np.sin(np.radians(self.currentPos))
            )

            netTorque = torque + noiseTorque

            angularAcceleration = netTorque / self.I
            self.currentPos += angularAcceleration * self.timeStep**2

            self.angles.append(self.currentPos)
            self.time.append(self.time[-1] + self.timeStep)
            self.n += 1

        plot_angles_over_time(self.time, self.angles)


def main():

    cntrl = Controller(50, 0.1, 5)
    sim = Simulation(cntrl, 9, 9.8, 0.2, 0.75, 0.5, 0.02, 0)
    sim.simulate(10, 8)
    
if __name__ == "__main__":
    main()
