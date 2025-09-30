from mtlsp.controller.vehicle_controller.controller_carla import Controller_Carla

class TMController(Controller_Carla):
    def __init__(self, observation_method=None, env=None, controller_type="TMController"):
        super().__init__(observation_method=observation_method, controllertype=controller_type)
        self.env = env
        self.control_log = {
            "weight_list_per_simulation": [1.0],
            "criticality": 0.0,
            "ndd_possi": 1.0,
            "discriminator_input": {}
        }
        self.drl_epsilon_value = -1
        self.real_epsilon_value = -1

    def reset(self):
        pass

    def attach_to_vehicle(self,vehicle_wrapper):
        super().attach_to_vehicle(vehicle_wrapper)

    def step(self):
        control = self.vehicle_wrapper.vehicle.get_control()
        self.action = {
            "throttle": control.throttle,
            "steer": control.steer,
            "brake": control.brake
        }
        self.ego_info =  self.vehicle_wrapper.observation.information["Ego"]
        return self.ego_info
