import os
import json
from pathlib import Path
from math import isclose
from functools import reduce
from mtlsp.logger.infoextractor import InfoExtractor
import copy
import conf.conf as conf


class CarlaInfoExtractor(InfoExtractor):
    def __init__(self, env):
        super().__init__(env)
        self.episode_log = {
            "collision_result": None,
            "collision_id": None,
            "weight_episode": 1,
            "current_weight": 1,
            "episode_info": None,
            "crash_decision_info": None,
            "decision_time_info": {},
            "weight_step_info": {},
            "drl_epsilon_step_info": {},
            "real_epsilon_step_info": {},
            "criticality_this_timestep": 0,
            "criticality_step_info": {},
            "ndd_step_info": {},
            "drl_obs_step_info": {},
            "ttc_step_info": {},
            "distance_step_info": {},
            "av_obs": {}
        }
        self.record = {}
        self.initial_log = {}
        self.weight_result = 0
        self.save_dir = self.env.experiment_path if hasattr(self.env, "experiment_path") else "./carla_experiments"

    def add_initialization_info(self, vehID, information):
        self.initial_log[vehID] = copy.deepcopy(information)

    from math import isclose
    from functools import reduce

    def get_snapshot_info(self, control_info=None):
        """Obtain the vehicle information at every time step."""
        self.save_dir = self.env.experiment_path
        time_step = round(self.env.get_simulation_time() - self.env.step_size, 2)

        mode = conf.experiment_config.get("mode", "NDE")
        controller = self.env.global_controller_instance_list[0]  # or get_controller_by_role("CAV")

        # [1] Get weight list
        snapshot_weight_list = controller.control_log.get("weight_list_per_simulation", [1.0]) \
            if mode in ["D2RL", "behavior_policy"] else [1.0]

        # [2] Update weight
        total_weight = reduce(lambda x, y: x * y, snapshot_weight_list)
        self.episode_log["weight_episode"] *= total_weight
        self.episode_log["current_weight"] = total_weight

        # [3] Distance / TTC
        try:
            distance, ttc = self.env.get_av_ttc()
            self.episode_log["distance_step_info"][time_step] = distance
            self.episode_log["ttc_step_info"][time_step] = ttc
        except Exception:
            pass

        try:
            self.episode_log["av_obs"][time_step] = self.env.get_av_obs()
        except Exception:
            pass

        # [4] Log criticality, DRL info
        try:
            self.episode_log["criticality_step_info"][time_step] = controller.control_log.get("criticality", 0.0)

            if not isclose(total_weight, 1.0):
                self.episode_log["weight_step_info"][time_step] = total_weight
                self.episode_log["drl_obs_step_info"][time_step] = self.get_current_drl_obs()

            if getattr(controller, "drl_epsilon_value", -1) != -1:
                self.episode_log["drl_epsilon_step_info"][time_step] = controller.drl_epsilon_value

            if getattr(controller, "real_epsilon_value", -1) != -1:
                self.episode_log["real_epsilon_step_info"][time_step] = controller.real_epsilon_value

            if "ndd_possi" in controller.control_log:
                self.episode_log["ndd_step_info"][time_step] = controller.control_log["ndd_possi"]

        except Exception as e:
            print("Log error:", e)

    def get_current_drl_obs(self):
        return self.env.global_controller_instance_list[0].control_log.get("discriminator_input", {})

    import os
    import json

    def get_terminate_info(self, stop, reason, additional_info):
        if conf.experiment_config["mode"] == "DRL_train":
            return

        if not stop:
            return

        self.episode_log["episode_info"] = self.env.episode_info
        self.episode_log["collision_result"] = 1 if self.env.check_collision() else 0
        self.episode_log["collision_id"] = additional_info.get("collision_id", [])
        self.episode_log["initial_criticality"] = 1
        self.episode_log["initial_weight"] = getattr(self.env, "initial_weight", 1)
        self.episode_log["reject_flag"] = False

        if 1 in reason and self.episode_log["collision_id"]:
            crash_decision_info = {}
            for vid in self.episode_log["collision_id"]:
                controller = self.env.get_controller_by_id(vid)
                if controller:
                    crash_decision_info[vid] = {
                        "ego_info": getattr(controller, "ego_info", {}),
                        "action": getattr(controller, "action", {})
                    }
            self.episode_log["crash_decision_info"] = crash_decision_info
        else:
            self.episode_log["crash_decision_info"] = None

        if not (1 in reason or conf.experiment_config["log_mode"] == "all"):
            self.episode_log["decision_time_info"] = None
            self.episode_log["crash_decision_info"] = None

        episode_id = self.episode_log["episode_info"]["id"]
        filename = f"episode_{episode_id:04d}.json"

        # Save to run_xxx/tested_and_safe or crash
        root_dir = self.env.experiment_path

        if 1 in reason:
            save_dir = os.path.join(root_dir, "crash")
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            with open(filepath, "w") as f:
                json.dump(self.episode_log, f, indent=2)
            self.weight_result = float(self.episode_log["weight_episode"])
        else:
            save_dir = os.path.join(root_dir, "tested_and_safe")
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            if self.meet_log_criteria(self.episode_log["ttc_step_info"], self.episode_log["distance_step_info"]):
                with open(filepath, "w") as f:
                    json.dump(self.episode_log, f, indent=2)
            else:
                crash_xml = os.path.join(root_dir, "crash", getattr(self.env, "output_filename", "") + ".fcd.xml")
                if os.path.isfile(crash_xml):
                    os.remove(crash_xml)
            self.weight_result = 0

        if 6 in reason:
            os.makedirs(os.path.join(root_dir, "rejected"), exist_ok=True)

        self.episode_log = {
            "collision_result": None,
            "collision_id": None,
            "weight_episode": 1,
            "current_weight": 1,
            "episode_info": None,
            "crash_decision_info": None,
            "decision_time_info": {},
            "weight_step_info": {},
            "drl_epsilon_step_info": {},
            "real_epsilon_step_info": {},
            "criticality_this_timestep": 0,
            "criticality_step_info": {},
            "ndd_step_info": {},
            "drl_obs_step_info": {},
            "ttc_step_info": {},
            "distance_step_info": {},
            "av_obs": {}
        }

    def meet_log_criteria(self, ttc_dict, distance_dict):
        min_distance, min_ttc = self.calculate_min_distance_ttc(ttc_dict, distance_dict)
        return (min_ttc < 5) or (min_distance < 10)

    def calculate_min_distance_ttc(self, ttc_dict, distance_dict):
        min_ttc = min(ttc_dict.values(), default=10000)
        min_distance = min(distance_dict.values(), default=10000)
        return min_distance, min_ttc


