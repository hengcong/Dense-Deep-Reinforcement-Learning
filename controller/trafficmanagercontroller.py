class TMController:
    def __init__(self, vehicle=None):
        self.vehicle = vehicle
        self.control_log = {
            "weight_list_per_simulation": [1.0],
            "criticality": 0.0,
            "ndd_possi": 1.0,
            "discriminator_input": {}
        }
        self.drl_epsilon_value = -1
        self.real_epsilon_value = -1

    def decision(self, obs, env=None):
        if self.vehicle is not None:
            control = self.vehicle.get_control()

            self.control_log["latest_control"] = {
                "throttle": control.throttle,
                "brake": control.brake,
                "steer": control.steer
            }
        else:
            self.control_log["latest_control"] = {
                "throttle": 0.0,
                "brake": 0.0,
                "steer": 0.0
            }

        return {
            "longitudinal": 0.0,
            "lateral": "still"
        }, self.control_log
