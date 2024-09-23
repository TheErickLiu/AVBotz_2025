import argparse
import numpy as np
import matplotlib.pyplot as plt

# define target position as global variable
TARGET_POSITION = [2, 3]


class MPPIController:
    """
    Model Predictive Control with Path Integral (MPPI) algorithm based on research paper
    "Model Predictive Path Integral Control using Covariance Variable Importance Sampling"
    (Algorithm 1, page 6, in https://arxiv.org/pdf/1509.01149).
    """
    def __init__(self, num_sample: int, horizon_steps: int, step_time: float, sigma: float, lamb: float):
        self.numSamples = num_sample       # sample size
        self.horizonSteps = horizon_steps  # number of steps in horizon
        self.stepTime = step_time
        self.sigma = sigma    # sigma determins the scale of control
        self.lamb = lamb      # lambda determins weights of rollouts

    # Dynamics of the vehicle
    def next_pos(self, x: float, y: float, control: list) -> tuple:
        # calculate next pos with determined velocity
        x_next = x + control[0] * self.stepTime
        y_next = y + control[1] * self.stepTime
        return x_next, y_next

    # Use a normal distribution for cost function
    def random_samples(self, center: float = 0.0) -> np.ndarray:
        return np.random.normal(center, self.sigma, size=(self.numSamples, self.horizonSteps, 2))

    # MPPI cost function to optimize. Currently it considers distance to the target
    # and scale of the control. 
    # 1. Higher control cost weight does not generate smooth plots. But I believe
    # it will smooth operations in the real environment. 
    # 2. I tried to penalize negative movements, but it didn't work well.
    def cost(self, x: float, y: float, ctrl: list, control_weight: float = 0.1) -> float:
        dist_cost = (x - TARGET_POSITION[0])**2 + (y - TARGET_POSITION[1])**2
        control_cost = ctrl[0]**2 + ctrl[1]**2
        return dist_cost + control_weight * control_cost

    def get_costs(self, control_samples: list, x0: float, y0: float) -> np.ndarray:
        # Rollout the trajectory of each sample
        costs = np.zeros(self.numSamples)
        for i in range(self.numSamples):
            x, y = x0, y0
            for j in range(self.horizonSteps):
                ctrl = control_samples[i, j]
                x, y = self.next_pos(x, y, ctrl)
                costs[i] += self.cost(x, y, ctrl) * self.stepTime
        return costs

    def calc_weights_from_costs(self, costs):
        weights = np.exp(-(costs) / self.lamb)
        weights /= np.sum(weights)
        return weights
        
    def compute_control(self, x0: float, y0: float):
        """Return MPPI control"""

        # Generate random control samples centered at 0
        controlSamples = self.random_samples()

        # Rollout the trajectory of each sample
        costs = self.get_costs(controlSamples, x0, y0)

        # Calcuate weights for costs
        weights = self.calc_weights_from_costs(costs)

        # Update control input with the first step in horizon
        ctrl_best = np.zeros(2)
        for j in range(self.numSamples):
            ctrl_best += weights[j] * controlSamples[j, 0]

        return ctrl_best
    

def plot_traj(x_traj: list, y_traj: list, cx_traj: list, cy_traj: list, times: list, filename: str = "positions.png"):
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 2, 1)
    plt.plot(times, x_traj)
    plt.axhline(y=TARGET_POSITION[0], color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time (s)')
    plt.ylabel('X Position')
    plt.title('X Position vs. Time')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(times, y_traj)
    plt.axhline(y=TARGET_POSITION[1], color='r', linestyle='--', label='Setpoint')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Position')
    plt.title('Y Position vs. Time')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(times, cx_traj)
    plt.xlabel('Time (s)')
    plt.ylabel('Control X')
    plt.title('Control X vs. Time')
    
    plt.subplot(2, 2, 4)
    plt.plot(times, cy_traj)
    plt.xlabel('Time (s)')
    plt.ylabel('Control Y')
    plt.title('Control Y vs. Time')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


class Simulation:
    def __init__(self, controller: MPPIController):
        self.controller = controller
        self.x_traj = []
        self.y_traj = []
        self.cx_traj = [0]
        self.cy_traj = [0]
        self.times = [0]

    def simulate(self, x: float, y: float, duration: float):
        self.x_traj.append(x)
        self.y_traj.append(y)
        
        total_steps = (int)(duration/self.controller.stepTime)

        for n in range(total_steps):
            control = self.controller.compute_control(x, y)
            self.cx_traj.append(control[0])
            self.cy_traj.append(control[1])

            x, y = self.controller.next_pos(x, y, control)
            self.x_traj.append(x)
            self.y_traj.append(y)

            self.times.append((n+1) * self.controller.stepTime)

        plot_traj(self.x_traj, self.y_traj, self.cx_traj, self.cy_traj, self.times)


def main():
    controller = MPPIController(100, 20, 0.1, 0.5, 0.1)
    sim = Simulation(controller)
    sim.simulate(0, 0, 20)

if __name__ == "__main__":
    main()
    