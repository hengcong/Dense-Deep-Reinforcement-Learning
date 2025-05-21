from mtlsp.observation.observation_carla import ObservationCarla
import numpy as np
import carla
class VehicleWrapper:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.controller = None
        self.controlled_flag = False
        self.observation = None  # Observation object if needed
        self.id = vehicle.id     # Alias for convenience
        self.role = None         # Optional: 'CAV' or 'BV' or 'Pedestrian'
        self.controlled_duration = 0

    def install_controller(self, controller):
        self.controller = controller
        if hasattr(self.controller, 'attach_to_vehicle'):
            controller.attach_to_vehicle(self)

    def reset_control_state(self):
        if self.controller and hasattr(self.controller, 'reset'):
            self.controller.reset()

    def is_action_legal(self,env, action):
        """
        Check whether the action is legal for this vehicle in the current CARLA lane topology.

        Args:
            action (dict): Action with keys "lateral" and "longitudinal", e.g., {"lateral": "left", "longitudinal": 0.5}

        Returns:
            bool: True if legal, False otherwise.
        """
        if "lateral" not in action:
            return False  # action missing key

        lateral_cmd = action["lateral"]

        # +1: check left lane, -1: check right lane
        if lateral_cmd == "left":
            return env.has_adjacent_lane(self.vehicle, direction=1)
        elif lateral_cmd == "right":
            return env.has_adjacent_lane(self.vehicle, direction=-1)
        else:
            return True  # "keep" or unknown → assume legal

    def step(self):
        if self.controller and hasattr(self.controller, 'step'):
            self.controller.step()

    def update(self, env):
        if self.controller and self.controller.action is not None:
            action = self.controller.action

            if self.is_action_legal(env, action):
                control = carla.VehicleControl()

                acc = action["longitudinal"]
                steer_cmd = action["lateral"]

                control.throttle = np.clip(acc / 3.0, 0, 1) if acc > 0 else 0
                control.brake = np.clip(-acc / 8.0, 0, 1) if acc <= 0 else 0
                control.steer = {"left": -0.3, "right": 0.3, "central": 0.0}[steer_cmd]

                self.vehicle.apply_control(control)

                # update flags
                if steer_cmd in ["left", "right"]:
                    self.controlled_duration += 1
                self.controlled_flag = True

    def set_role(self, role_name):
        self.role = role_name

    def set_observation(self, obs):
        self.observation = obs

    def update_observation(self, env, time_stamp=None):
        if time_stamp is None:
            time_stamp = env.get_simulation_time()
        obs = ObservationCarla(veh_id=self.vehicle.id, time_stamp=time_stamp)
        obs.update(env)
        self.set_observation(obs)

