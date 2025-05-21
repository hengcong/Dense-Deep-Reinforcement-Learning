from mtlsp.controller.vehicle_controller.globalcontroller import DummyGlobalController
from controller.treesearchcarlacontroller import TreeSearchController
import numpy as np
from copy import deepcopy
import collections
import utils
from conf import conf
import carla

class TreeSearchBVGlobalController(DummyGlobalController):
    controlled_bv_num = 4 # NOTE: not sure
    def __init__(self, env, veh_type="BV"):
        super().__init__(env, veh_type)
        self.control_vehicle_set = set()
        self.drl_info =None
        self.drl_epsilon_value = -1
        self.real_epsilon_value = -1

    def step(self):
        """
        Complete TreeSearch Controller Step Logic
        This function performs a simulation step including:
        - Resetting control/action states
        - Updating controllable vehicle list
        - Executing default NDE control logic
        - Optionally executing NADE (or D2RL) override logic
        - Returning the vehicle-level criticality list (for logging/evaluation)
        """
        self.real_epsilon_value = -1
        self.drl_epsilon_value = -1
        self.control_log = {"criticality": 0, "discriminator_input": 0}

        # Buffers for NADE logging or decision
        bv_action_idx_list = []
        weight_list = []
        max_vehicle_criticality = []
        ndd_possi_list = []
        IS_possi_list = []
        controlled_bvs_list = []
        vehicle_criticality_list = []

        # === Reset vehicle control state ===
        self.reset_control_and_action_state()

        # === Refresh list of vehicles under consideration for control ===
        self.update_controlled_vehicles(controller=TreeSearchController)

        # === NDE Default Control Phase ===
        if conf.experiment_config["mode"] == "NDE":
            for bv_id in self.controllable_veh_id_list:
                bv = self.env.vehicle_wrapper_list[bv_id]

                # Step the controller (computes action)
                bv.controller.step()

                # Apply the action from controller to vehicle
                bv.update(self.env)

        # === NADE / D2RL Adversarial Control Phase ===
        elif conf.experiment_config["mode"] in ["D2RL", "behavior_policy"]:

            # Phase 1: Determine control action using TreeSearch logic
            bv_action_idx_list, weight_list, max_vehicle_criticality, \
                ndd_possi_list, IS_possi_list, controlled_bvs_list, \
                vehicle_criticality_list, _ = self.select_controlled_bv_and_action()

            # Phase 2: Apply selected actions
            for bv_id in self.controllable_veh_id_list:
                bv = self.env.vehicle_wrapper_list[bv_id]

                if bv in controlled_bvs_list:
                    nade_action = bv_action_idx_list[controlled_bvs_list.index(bv)]

                    if nade_action is not None:
                        self.control_log["ndd_possi"] = ndd_possi_list[controlled_bvs_list.index(bv)]

                        # Override action to TreeSearch-based selected action
                        bv.controller.action = utils.action_id_to_action_command(nade_action)
                        bv.controller.NADE_flag = True

                        # Optional: visualize as blue for adversarial BV
                        bv.simulator.set_vehicle_color(bv.vehicle.id, bv.color_blue)

                # Apply controller's selected action
                bv.update()

            # Record weights for logging
            self.control_log["weight_list_per_simulation"] = [val for val in weight_list if val is not None]
            if len(self.control_log["weight_list_per_simulation"]) == 0:
                self.control_log["weight_list_per_simulation"] = [1]

        else:
            raise ValueError(f"[TreeSearchController] Unsupported mode: {conf.experiment_config['mode']}")

        # === Return criticality list for logging/evaluation ===
        return vehicle_criticality_list

    def _get_controllable_veh_id_list(self):
        return [v.id for v in self.env.vehicles]

    def reset_control_and_action_state(self):
        """Reset control state of autonomous vehicles.
        """
        for veh_id in self.controllable_veh_id_list:
            vehicle = self.env.vehicle_wrapper_list[veh_id]
            vehicle.reset_control_state()

    def update_controlled_vehicles(self, controller=TreeSearchController):
        env = self.env
        CAV = env.ego_vehicle_wrapper

        context_vehicle_set = set(CAV.observation.context.keys())

        if self.control_vehicle_set != context_vehicle_set:
            for veh_id in context_vehicle_set - self.control_vehicle_set:
                env.get_surrounding_vehicles(env.vehicle_wrapper_list[veh_id].vehicle)
                ctrl = controller(env=env)
                ctrl.attach_to_vehicle(env.vehicle_wrapper_list[veh_id])
                env.vehicle_wrapper_list[veh_id].install_controller(ctrl)

            for veh_id in self.control_vehicle_set - context_vehicle_set:
                if veh_id in env.vehicle_wrapper_list:
                    env.vehicle_wrapper_list[veh_id].install_controller(None)

            self.control_vehicle_set = context_vehicle_set

    def apply_control_permission(self):
        for vehicle in self.get_bv_candidates():
            obs_ego = vehicle.observation.information.get("Ego", {})
            if vehicle.controller.NADE_flag and utils.is_lane_change(obs_ego):
                return False
        return True

    def get_bv_candidates(self):
        av_id = self.env.ego_vehicle.id
        av_obs_info = self.env.step_data[av_id]["observation"]
        av_pos = av_obs_info["Ego"][66]
        av_context = av_obs_info["context"]

        bv_list = []
        for bv_id, bv_obs in av_context.items():
            bv_pos = bv_obs[66]
            dist = utils.cal_euclidean_dist(av_pos, bv_pos)
            if dist <= conf.cav_obs_range:
                bv_list.append((bv_id, dist))

        bv_list.sort(key=lambda x: x[1])
        candidates = [
            self.env.vehicle_wrapper_list[str(bv_id)]
            for bv_id, _ in bv_list[:self.controlled_bv_num]
        ]
        return candidates

    def run_adversarial_planning_and_control(self): # NOTE: New Function
        bv_action_idx_list = []
        weight_list = []
        max_vehicle_criticality = []
        ndd_possi_list = []
        IS_possi_list = []
        controlled_bvs_list = []
        vehicle_criticality_list = []

        bv_action_idx_list, weight_list, max_vehicle_criticality, \
            ndd_possi_list, IS_possi_list, controlled_bvs_list, \
            vehicle_criticality_list, _ = self.select_controlled_bv_and_action()

        for bv_id in self.controllable_veh_id_list:
            bv = self.env.vehicle_wrapper_list[bv_id]
            if bv in controlled_bvs_list:
                nade_action = bv_action_idx_list[controlled_bvs_list.index(bv)]
                if nade_action is not None:
                    self.control_log["ndd_possi"] = ndd_possi_list[controlled_bvs_list.index(bv)]
                    bv.controller.action = utils.action_id_to_action_command(nade_action)
                    bv.controller.NADE_flag = True
                    if hasattr(bv, "simulator"):
                        bv.simulator.set_vehicle_color(bv.vehicle.id, bv.color_blue)
            bv.update()  # apply bv.controller.action

        self.control_log["weight_list_per_simulation"] = [val for val in weight_list if val is not None]
        if len(self.control_log["weight_list_per_simulation"]) == 0:
            self.control_log["weight_list_per_simulation"] = [1]

        return vehicle_criticality_list

    def select_controlled_bv_and_action(self):
        """Select background vehicles and compute actions via TreeSearch controller."""
        num_controlled_critical_bvs = 1
        controlled_bvs_list = self.get_bv_candidates()
        CAV_obs = self.env.ego_vehicle_wrapper.observation.information
        full_obs = self.get_full_obs_from_cav_obs_and_bv_list(CAV_obs, controlled_bvs_list)
        self.nade_candidates = controlled_bvs_list
        bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list = self.calculate_criticality_list(
            controlled_bvs_list, CAV_obs, full_obs)
        whole_weight_list = []
        self.control_log["criticality"] = sum(bv_criticality_list)

        discriminator_input = self.collect_discriminator_input_simplified(full_obs, controlled_bvs_list,
                                                                          bv_criticality_list)
        self.control_log["discriminator_input"] = discriminator_input.tolist()
        self.epsilon_value = -1
        underline_drl_action = self.get_underline_drl_action(discriminator_input, bv_criticality_list)

        if sum(bv_criticality_list) > 0:
            self.drl_epsilon_value = underline_drl_action
            self.real_epsilon_value = underline_drl_action

        for i in range(len(controlled_bvs_list)):
            bv = controlled_bvs_list[i]
            bv_criticality = bv_criticality_list[i]
            bv_criticality_array = criticality_array_list[i]
            bv_pdf = bv.controller.get_NDD_possi()
            combined_bv_criticality_array = bv_criticality_array
            bv_action_idx, weight, ndd_possi, critical_possi, single_weight_list = bv.controller.Decompose_sample_action(
                np.sum(combined_bv_criticality_array), combined_bv_criticality_array, bv_pdf, underline_drl_action)
            if bv_action_idx is not None:
                bv_action_idx = bv_action_idx.item()
            bv_action_idx_list.append(bv_action_idx)
            weight_list.append(weight)
            ndd_possi_list.append(ndd_possi)
            IS_possi_list.append(critical_possi)
            if single_weight_list is not None:
                whole_weight_list.append(min(single_weight_list))
            else:
                whole_weight_list.append(None)

        vehicle_criticality_list = deepcopy(bv_criticality_list)
        selected_bv_idx = sorted(range(len(bv_criticality_list)), key=lambda i: bv_criticality_list[i])[
                          -num_controlled_critical_bvs:]
        for i in range(len(controlled_bvs_list)):
            if i in selected_bv_idx:
                if whole_weight_list[i] and whole_weight_list[i] * self.env.info_extractor.episode_log[
                    "weight_episode"] * self.env.initial_weight < conf.weight_threshold:
                    bv_action_idx_list[i] = weight_list[i] = ndd_possi_list[i] = IS_possi_list[i] = None
            else:
                bv_action_idx_list[i] = weight_list[i] = ndd_possi_list[i] = IS_possi_list[i] = None

        max_vehicle_criticality = np.max(bv_criticality_list) if len(bv_criticality_list) else -np.inf

        return bv_action_idx_list, weight_list, max_vehicle_criticality, ndd_possi_list, IS_possi_list, controlled_bvs_list, vehicle_criticality_list, discriminator_input

    @staticmethod
    def pre_load_predicted_obs_and_traj(full_obs):
        predicted_obs = {}
        trajectory_obs = {}
        action_list = ["left", "right", "still"]
        for veh_id in full_obs:
            for action in action_list:
                vehicle = full_obs[veh_id]
                if veh_id not in trajectory_obs:
                    trajectory_obs[veh_id] = {}
                if veh_id not in predicted_obs:
                    predicted_obs[veh_id] = {}
                predicted_obs[veh_id][action], trajectory_obs[veh_id][action] = TreeSearchController.update_single_vehicle_obs(vehicle, action)
        return predicted_obs, trajectory_obs

    def _get_Surrogate_CAV_action_probability(self):
        """Predict the action probability of the ego vehicle based on a surrogate model"""
        CAV_left_prob, CAV_right_prob = 0, 0
        CAV_still_prob = conf.epsilon_still_prob
        left_gain, right_gain = 0, 0
        left_LC_safety_flag, right_LC_safety_flag = False, False

        ego_vehicle = self.env.ego_vehicle
        current_wp = self.env.map.get_waypoint(
            ego_vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if current_wp is None:
            return 0.0, 1.0, 0.0  # fallback: only still

        left_wp = current_wp.get_left_lane()
        right_wp = current_wp.get_right_lane()

        # Evaluate left lane
        if left_wp and left_wp.lane_type == carla.LaneType.Driving and left_wp.lane_id * current_wp.lane_id > 0:
            LC_safety_flag, gain = self._Mobil_surrogate_model(ego_vehicle, left_wp)
            if gain is not None:
                left_gain = np.clip(gain, 0., None)
                left_LC_safety_flag = LC_safety_flag

        # Evaluate right lane
        if right_wp and right_wp.lane_type == carla.LaneType.Driving and right_wp.lane_id * current_wp.lane_id > 0:
            LC_safety_flag, gain = self._Mobil_surrogate_model(ego_vehicle, right_wp)
            if gain is not None:
                right_gain = np.clip(gain, 0., None)
                right_LC_safety_flag = LC_safety_flag

        assert left_gain >= 0 and right_gain >= 0

        CAV_left_prob += conf.epsilon_lane_change_prob * left_LC_safety_flag
        CAV_right_prob += conf.epsilon_lane_change_prob * right_LC_safety_flag

        max_remaining_LC_prob = 1 - conf.epsilon_still_prob - CAV_left_prob - CAV_right_prob
        total_gain = left_gain + right_gain

        obtained_LC_prob_for_sharing = np.clip(
            utils.remap(total_gain, [0, conf.SM_MOBIL_max_gain_threshold], [0, max_remaining_LC_prob]),
            0, max_remaining_LC_prob
        )

        CAV_still_prob += (max_remaining_LC_prob - obtained_LC_prob_for_sharing)

        if total_gain > 0:
            CAV_left_prob += obtained_LC_prob_for_sharing * (left_gain / total_gain)
            CAV_right_prob += obtained_LC_prob_for_sharing * (right_gain / total_gain)

        assert 0.99999 <= (CAV_left_prob + CAV_still_prob + CAV_right_prob) <= 1.0001

        return CAV_left_prob, CAV_still_prob, CAV_right_prob

    def collect_discriminator_input_simplified(self, full_obs, controlled_bvs_list, bv_criticality_list):
        """
        Construct D2RL input observation from full observation and critical BVs.

        Args:
            full_obs (OrderedDict): Observation dictionary with "CAV" and BV ids.
            controlled_bvs_list (List[VehicleWrapper]): List of selected BV wrappers.
            bv_criticality_list (List[float]): Criticality scores for selected BVs.

        Returns:
            np.ndarray: Normalized observation vector for discriminator.
        """
        # === CAV state ===
        CAV_global_position = list(full_obs["CAV"]["position"])  # (x, y)
        CAV_speed = full_obs["CAV"]["speed"]

        # === episode weight ===
        tmp_weight = self.env.info_extractor.episode_log.get("weight_episode", 1.0)
        tmp_weight = np.log10(tmp_weight)

        # === Controlled BV info ===
        vehicle_info_list = []
        controlled_bv_num = 1
        total_bv_info_length = controlled_bv_num * 4

        if bv_criticality_list: # NOTE: doublecheck with this list
            selected_bv_index = int(np.argmax(np.array(bv_criticality_list)))
            vehicle = controlled_bvs_list[selected_bv_index]
            veh_id = vehicle.id
            vehicle_single_obs = full_obs[veh_id]

            vehicle_position = list(vehicle_single_obs["position"])
            vehicle_relative_position = [
                vehicle_position[0] - CAV_global_position[0],
                vehicle_position[1] - CAV_global_position[1]
            ]
            vehicle_relative_speed = vehicle_single_obs["speed"] - CAV_speed
            predict_relative_position = vehicle_relative_position[0] + vehicle_relative_speed

            vehicle_info_list.extend(
                vehicle_relative_position + [vehicle_relative_speed, predict_relative_position]
            )
        else:
            vehicle_info_list.extend([-20, -8, -10, -20])

        if len(vehicle_info_list) < total_bv_info_length:
            vehicle_info_list.extend([-1] * (total_bv_info_length - len(vehicle_info_list)))

        # === Criticality flags ===
        bv_criticality_flag = 1 if sum(bv_criticality_list) > 0 else 0
        bv_criticality_value = np.log10(sum(bv_criticality_list)) if sum(bv_criticality_list) > 0 else 16

        # === Normalization bounds ===
        if conf.simulation_config["map"] == "2LaneLong":
            CAV_position_lb, CAV_position_ub = [400, 40], [4400, 50]
        else:
            CAV_position_lb, CAV_position_ub = [400, 40], [800, 50]

        CAV_velocity_lb, CAV_velocity_ub = 0, 20
        weight_lb, weight_ub = -30, 0
        bv_criticality_flag_lb, bv_criticality_flag_ub = 0, 1
        bv_criticality_value_lb, bv_criticality_value_ub = -16, 0
        vehicle_info_lb, vehicle_info_ub = [-20, -8, -10, -20], [20, 8, 10, 20]

        lb_array = np.array(
            CAV_position_lb + [CAV_velocity_lb] + [weight_lb] + [bv_criticality_flag_lb] + [bv_criticality_value_lb] +
            vehicle_info_lb * controlled_bv_num
        )
        ub_array = np.array(
            CAV_position_ub + [CAV_velocity_ub] + [weight_ub] + [bv_criticality_flag_ub] + [bv_criticality_value_ub] +
            vehicle_info_ub * controlled_bv_num
        )

        # === Observation vector ===
        total_obs_ori = np.array(
            CAV_global_position + [CAV_speed] + [tmp_weight] + [bv_criticality_flag] + [bv_criticality_value] + vehicle_info_list
        )
        total_obs_norm = 2 * (total_obs_ori - lb_array) / (ub_array - lb_array) - 1
        total_obs_clipped = np.clip(total_obs_norm, -5, 5)

        return np.float32(total_obs_clipped)

    def calculate_criticality_list(self, controlled_bvs_list, CAV_obs, full_obs):
        # Initialize return lists
        bv_criticality_list = []
        criticality_array_list = []
        bv_action_idx_list = []
        weight_list = []
        ndd_possi_list = []
        IS_possi_list = []

        predicted_full_obs, predicted_traj_obs = TreeSearchBVGlobalController.pre_load_predicted_obs_and_traj(full_obs) # NOTE: @staticmthod, defined but not tested yet
        cav_obs_info = self.env.ego_vehicle_wrapper.observation.information
        CAV_left_prob, CAV_still_prob, CAV_right_prob = self._get_Surrogate_CAV_action_probability(
            cav_obs=cav_obs_info
        ) # NOTE: not defined yet

        for bv in controlled_bvs_list:
            bv_criticality, criticality_array =bv.controller.Decompose_decision(
                CAV_obs,
                SM_LC_prob=[CAV_left_prob, CAV_still_prob, CAV_right_prob],
                full_obs=full_obs,
                predicted_full_obs=predicted_full_obs,
                predicted_traj_obs=predicted_traj_obs
            )
            bv_criticality_list.append(bv_criticality)
            criticality_array_list.append(criticality_array)

        return bv_criticality_list, criticality_array_list, bv_action_idx_list, weight_list, ndd_possi_list, IS_possi_list


    def get_underline_drl_action(self, discriminator_input, bv_criticality_list):
        underline_drl_action = None

        if sum(bv_criticality_list) > 0:
            setting = conf.simulation_config.get("epsilon_setting", "fixed")
            if conf.simulation_config["epsilon_setting"] == "drl":
                if conf.discriminator_agent is None:
                    conf.discriminator_agent = conf.load_discriminator_agent() #NOTE: mode="torch", checkpoint_path=... only recommend torch in CARLA

                underline_drl_action = conf.discriminator_agent.compute_action(discriminator_input)
                underline_drl_action = max(0, min(underline_drl_action, 1))

                print(underline_drl_action, self.env.info_extractor.episode_log["weight_episode"])

            elif conf.simulation_config["epsilon_setting"] == "fixed":
                underline_drl_action = conf.epsilon_value
        return underline_drl_action

    @staticmethod
    def get_Surrigate_CAV_action_probability(cav_obs):
        CAV_left_prob, CAV_right_prob = 0, 0
        CAV_still_prob = conf.epsilon_still_prob
        left_gain, right_gain = 0, 0
        left_LC_safety_flag, right_LC_safety_flag = False, False
        lane_index_list = [-1, 1]  # -1: right turn; 1: left turn
        for lane_index in lane_index_list:
            LC_safety_flag, gain = TreeSearchBVGlobalController._Mobil_surraget_model(cav_obs, lane_index)
            if gain is not None:
                if lane_index == -1:
                    right_gain = np.clip(gain, 0., None)
                    right_LC_safety_flag = LC_safety_flag
                elif lane_index == 1:
                    left_gain = np.clip(gain, 0., None)
                    left_LC_safety_flag = LC_safety_flag
                assert (left_gain >= 0 and right_gain >= 0)

        CAV_left_prob += conf.epsilon_lane_change_prob * left_LC_safety_flag
        CAV_right_prob += conf.epsilon_lane_change_prob * right_LC_safety_flag

        max_remaining_LC_prob = 1 - conf.epsilon_still_prob - CAV_left_prob - CAV_right_prob

        total_gain = left_gain + right_gain
        obtained_LC_prob_for_sharing = np.clip(utils.remap(total_gain, [0, conf.SM_MOBIL_max_gain_threshold], [
                                               0, max_remaining_LC_prob]), 0, max_remaining_LC_prob)
        CAV_still_prob += (max_remaining_LC_prob -
                           obtained_LC_prob_for_sharing)

        if total_gain > 0:
            CAV_left_prob += obtained_LC_prob_for_sharing * \
                (left_gain/(left_gain + right_gain))
            CAV_right_prob += obtained_LC_prob_for_sharing * \
                (right_gain/(left_gain + right_gain))

        assert(0.99999 <= (CAV_left_prob + CAV_still_prob + CAV_right_prob) <= 1.0001)
        return CAV_left_prob, CAV_still_prob, CAV_right_prob

    def _Mobil_surrogate_model(self, cav_vehicle, surrounding: dict, target_lane: str):
        """
        MOBIL surrogate model for CAV lane change decision in CARLA.

        Args:
            cav_vehicle (carla.Vehicle): The ego vehicle.
            surrounding (dict): Dictionary of nearby vehicles, containing keys:
                'Lead', 'LeftLead', 'RightLead', 'Foll', 'LeftFoll', 'RightFoll'.
            target_lane (str): One of 'left' or 'right'.

        Returns:
            Tuple[bool, float or None]: (is_safe, gain)
        """
        gain = None
        cav_info = self._get_vehicle_observation(cav_vehicle, self.env.world)

        if target_lane == 'left':
            new_preceding = surrounding.get("LeftLead", None)
            new_following = surrounding.get("LeftFoll", None)
        elif target_lane == 'right':
            new_preceding = surrounding.get("RightLead", None)
            new_following = surrounding.get("RightFoll", None)
        else:
            return False, None

        r_new_preceding = 99999
        r_new_following = 99999
        if new_preceding:
            new_preceding_info = self._get_vehicle_observation(new_preceding, self.env.world)
            r_new_preceding = new_preceding_info["distance"]
        else:
            new_preceding_info = None
        if new_following:
            new_following_info = self._get_vehicle_observation(new_following, self.env.world)
            r_new_following = new_following_info["distance"]
        else:
            new_following_info = None

        if r_new_preceding <= 0 or r_new_following <= 0:
            return False, gain

        new_following_a = utils.acceleration(new_following_info, new_preceding_info)
        new_following_pred_a = utils.acceleration(new_following_info, cav_info)

        old_preceding = surrounding.get("Lead", None)
        old_following = surrounding.get("Foll", None)
        old_preceding_info = self._get_vehicle_observation(old_preceding, self.env.world) if old_preceding else None
        old_following_info = self._get_vehicle_observation(old_following, self.env.world) if old_following else None

        self_pred_a = utils.acceleration(cav_info, new_preceding_info)

        if new_following_pred_a < -conf.Surrogate_LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return True, 0

        self_a = utils.acceleration(cav_info, old_preceding_info)
        old_following_a = utils.acceleration(old_following_info, cav_info)
        old_following_pred_a = utils.acceleration(old_following_info, old_preceding_info)

        gain = self_pred_a - self_a + conf.Surrogate_POLITENESS * (
                new_following_pred_a - new_following_a +
                old_following_pred_a - old_following_a
        )

        return True, gain

    def get_full_obs_from_cav_obs_and_bv_list(self, CAV_obs, bv_list):
        """
        Construct full observation dict including CAV and selected BVs.

        Args:
            cav_wrapper: VehicleWrapper of the ego (CAV).
            bv_wrappers: List of VehicleWrapper for selected background vehicles.

        Returns:
            OrderedDict: Mapping from vehicle ID (e.g. "CAV", "BV_123") to its observation.
        """
        full_obs = collections.OrderedDict()
        full_obs["CAV"] = CAV_obs
        vehicle_id_list = [bv.vehicle.id for bv in bv_list]
        cav_surrounding = self._process_cav_context(vehicle_id_list)
        av_pos = CAV_obs["Ego"]["position"] # NOTE: not used
        for bv in bv_list:
            vehicle_id = bv.observation.information["Ego"]["veh_id"]
            full_obs[vehicle_id] = bv.observation.information["Ego"]
            bv_pos = bv.observation.information["Ego"]["position"] # NOTE: not used

        return full_obs

    def _process_cav_context(self, vehicle_id_list):
        def _process_cav_context(self, vehicle_id_list):
            """fetch information of all bvs from the cav context information
            """
            cav = self.env.vehicle_wrapper_list["CAV"]
            cav_pos = cav.observation.local["CAV"][66]
            cav_context = cav.observation.context
            cav_surrounding = {}
            cav_surrounding["CAV"] = {
                "range": 0,
                "lane_width": self.env.simulator.get_vehicle_lane_width("CAV"),
                "lateral_offset": cav.observation.local["CAV"][184],
                "lateral_speed": cav.observation.local["CAV"][50],
                "position": cav_pos,
                "prev_action": cav.observation.information["Ego"]["prev_action"],
                "relative_lane_index": 0,
                "speed": cav.observation.local["CAV"][64]
            }
            total_vehicle_id_list = list(
                set(vehicle_id_list) | set(cav_context.keys()))
            for veh_id in total_vehicle_id_list:
                bv_pos = cav_context[veh_id][66]
                distance = self.env.simulator.get_vehicles_dist_road("CAV", veh_id)

                if distance > conf.cav_obs_range + 5:
                    distance_alter = self.env.simulator.get_vehicles_dist_road(
                        veh_id, "CAV")
                    if distance_alter > conf.cav_obs_range + 5:
                        continue
                    else:
                        distance = -distance_alter
                        relative_lane_index = - \
                            self.env.simulator.get_vehicles_relative_lane_index(
                                veh_id, "CAV")
                else:
                    relative_lane_index = self.env.simulator.get_vehicles_relative_lane_index(
                        "CAV", veh_id)
                cav_surrounding[veh_id] = {
                    "range": distance,
                    "lane_width": self.env.simulator.get_vehicle_lane_width(veh_id),
                    "lateral_offset": cav_context[veh_id][184],
                    "lateral_speed": cav_context[veh_id][50],
                    "position": bv_pos,
                    "prev_action": self.env.vehicle_list[veh_id].observation.information["Ego"]["prev_action"],
                    "relative_lane_index": relative_lane_index,
                    "speed": cav_context[veh_id][64]
                }
            return cav_surrounding