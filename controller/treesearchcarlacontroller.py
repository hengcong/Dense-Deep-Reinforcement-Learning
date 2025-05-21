from collections import OrderedDict

import numpy as np
from bidict import bidict
import math
import scipy
import bisect
import utils # TODO: some functions need to be revised to adapt to the CARLA
import conf.conf as conf
from conf.defaultconf import CAV_acc_to_idx_dic, acc_to_idx_dic
from mtlsp.controller.vehicle_controller.idmcontroller import IDMController # TODO: replace with CARLA simulator
from controller.traj_predictor import Traj

class TreeSearchController:
    # Basic vehicle dynamic and discretization parameters
    longi_safety_buffer, lateral_safety_buffer = 2, 2
    v_low, v_high, r_low, r_high, rr_low, rr_high, acc_low, acc_high = 20, 40, 0, 115, -10, 8, -4, 2
    acc_resolution = 0.2
    LENGTH = 5
    speed_lb = conf.v_low
    speed_ub = conf.v_high
    LANE_CHANGE_INDEX_LIST = [0, 1, 2]
    ACTION_STEP = 1.0

    num_acc = int(1 + ((acc_high - acc_low) / acc_resolution))
    CAV_acc_low, CAV_acc_high, CAV_acc_step = -4, 2, 0.2
    num_CAV_acc = int((CAV_acc_high - CAV_acc_low) / CAV_acc_step + 1)

    # Build acceleration index dictionaries for BV and CAV
    CAV_acc_to_idx_dic = bidict()
    for i in range(num_CAV_acc):
        CAV_acc_to_idx_dic[list(np.arange(CAV_acc_low, CAV_acc_high + CAV_acc_step, CAV_acc_step))[i]] = i

    acc_to_idx_dic = bidict()
    for m in range(num_acc):
        acc_to_idx_dic[list(np.linspace(acc_low, acc_high, num=num_acc))[m]] = m
    """ Parameters from Tree Search Controller"""
    MAX_TREE_SEARCH_DEPTH = conf.treesearch_config["search_depth"]
    ACTION_NUM = 33  # full actions
    ACTION_TYPE = {"left": 0, "right": 1, "still": list(range(2, 33))}
    input_lower_bound = [-50, 20, 0] * 9
    input_lower_bound[0] = 400
    input_upper_bound = [50, 40, 2] * 9
    input_upper_bound[0] = 800
    input_lower_bound = np.array(input_lower_bound)
    input_upper_bound = np.array(input_upper_bound)
    if conf.treesearch_config["surrogate_model"] == "surrogate":
        SURROGATE_MODEL_FUNCTION = utils._get_Surrogate_CAV_action_probability
    elif conf.treesearch_config["surrogate_model"] == "AVI":
        SURROGATE_MODEL_FUNCTION = IDMController.decision_pdf #TODO
    PREDICT_MODEL_FUNCTION = IDMController.decision #TODO
    # lane_list = np.array([42.0, 46.0, 50.0])
    lane_list = conf.lane_list

    def __init__(self, env= None, observation_method=None, subscription_method=None, controllertype="NADEBackgroundController"):
        self.env = env
        self.vehicle = None # TODO:what does this vehicle refer to
        self.vehicle_wrapper = None # TODO: `vehicle` kept for legacy use; `vehicle_wrapper` added later for SUMO-style abstraction compatibility.

        self.ego_info = None
        self.action = None
        self._type = controllertype
        self.observation_method = observation_method
        self.control_log = {}
        self._recent_ndd_pdf = {"time_step": None, "pdf": None}
        self.NDD_flag = True
        self.NADE_flag = False
        self.map = None # TODO: CARLA map

        # class attribute from NADEBackgroundController
        self.weight = None
        self.ndd_possi = None
        self.critical_possi = None
        self.epsilon_pdf_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.normalized_critical_pdf_array = np.zeros(
            len(conf.ACTIONS), dtype=float)
        self.ndd_possi_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.bv_criticality_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.bv_challenge_array = np.zeros(len(conf.ACTIONS), dtype=float)

    def reset(self):
        self.NDD_flag, self.NADE_flag = True, False

    #TODO: New function different from SUMO code, make sure it works well.
    def attach_to_vehicle(self, vehicle_wrapper):
        self.NDD_flag, self.NADE_flag = True, False
        self.vehicle_wrapper = vehicle_wrapper
        self.vehicle = vehicle_wrapper.vehicle

    # TODO: vehicle in the carla environment is not defined with observation
    def step(self):
        self.ego_info = self.vehicle.observation.information["Ego"]


    """ 
    Functions from BaseController:  
        _check_longitudinal_safety()
        _check_lateral_safety()
    """

    @staticmethod
    def _check_longitudinal_safety(obs, pdf_array, lateral_result=None, CAV_flag=False): # NOTE:Although CAV_flag is set to False by default, no actual value is passed in during subsequent calls, which means all vehicles are treated as background vehicles (BV), even when they are actually CAVs.
        """Check longitudinal safety feasibility based on relative speed and distance."""
        ego_info = obs["Ego"]
        f_veh_info = obs["Lead"]
        safety_buffer = TreeSearchController.longi_safety_buffer

        for i in range(len(pdf_array) - 1, -1, -1):
            if CAV_flag:
                acc = TreeSearchController.CAV_acc_to_idx_dic.inverse[i]
            else:
                acc = TreeSearchController.acc_to_idx_dic.inverse[i]

            if f_veh_info is not None:
                rr = f_veh_info["speed"] - ego_info["speed"]
                r = f_veh_info["distance"]
                criterion_1 = rr + r + 0.5 * (TreeSearchController.acc_low - acc)

                self_v_2 = max(ego_info["speed"] + acc, TreeSearchController.v_low)
                f_v_2 = max(f_veh_info["speed"] + TreeSearchController.acc_low, TreeSearchController.v_low)

                dist_r = (self_v_2 ** 2 - TreeSearchController.v_low ** 2) / (2 * abs(TreeSearchController.acc_low))
                dist_f = (
                        (f_v_2 ** 2 - TreeSearchController.v_low ** 2) / (2 * abs(TreeSearchController.acc_low))
                        + TreeSearchController.v_low * (f_v_2 - self_v_2) / TreeSearchController.acc_low
                )

                criterion_2 = criterion_1 - dist_r + dist_f

                if criterion_1 <= safety_buffer or criterion_2 <= safety_buffer:
                    pdf_array[i] = 0
                else:
                    break

        lateral_feasible = (lateral_result[0] or lateral_result[2]) if lateral_result is not None else False

        if np.sum(pdf_array) == 0 and not lateral_feasible:
            pdf_array[0] = 1
            return pdf_array

        if CAV_flag:
            return pdf_array
        else:
            return pdf_array / np.sum(pdf_array)

    def _check_lateral_safety(self, obs, pdf_array):
        """Check lateral safety feasibility for lane changes."""
        ego_info = obs["Ego"]
        surrounding = {
            "LeftLead": obs["LeftLead"],
            "LeftFoll": obs["LeftFoll"],
            "RightLead": obs["RightLead"],
            "RightFoll": obs["RightFoll"]
        }
        safety_buffer = TreeSearchController.lateral_safety_buffer

        if not ego_info.get("could_drive_adjacent_lane_right", True):
            pdf_array[2] = 0
        if not ego_info.get("could_drive_adjacent_lane_left", True):
            pdf_array[0] = 0

        lane_change_dir = [0, 2]
        nearby_vehs = [[surrounding["LeftLead"], surrounding["LeftFoll"]], [surrounding["RightLead"], surrounding["RightFoll"]]]

        for lane_index, nearby in zip(lane_change_dir, nearby_vehs):
            if pdf_array[lane_index] != 0:
                f_veh, r_veh = nearby

                if f_veh is not None:
                    rr = f_veh["speed"] - ego_info["speed"]
                    r = f_veh["distance"]
                    dis_change = rr * TreeSearchController.ACTION_STEP + 0.5 * TreeSearchController.acc_low * (TreeSearchController.ACTION_STEP ** 2)
                    r_1 = r + dis_change
                    rr_1 = rr + TreeSearchController.acc_low * TreeSearchController.ACTION_STEP

                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0
                    elif rr_1 < 0:
                        self_v_2 = max(ego_info["speed"], TreeSearchController.v_low)
                        f_v_2 = max(f_veh["speed"] + TreeSearchController.acc_low, TreeSearchController.v_low)
                        dist_r = (self_v_2 ** 2 - TreeSearchController.v_low ** 2) / (2 * abs(TreeSearchController.acc_low))
                        dist_f = (f_v_2 ** 2 - TreeSearchController.v_low ** 2) / (2 * abs(TreeSearchController.acc_low)) + TreeSearchController.v_low * (f_v_2 - self_v_2) / TreeSearchController.acc_low
                        r_2 = r_1 - dist_r + dist_f
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0

                if r_veh is not None:
                    rr = ego_info["speed"] - r_veh["speed"]
                    r = r_veh["distance"]
                    dis_change = rr * TreeSearchController.ACTION_STEP - 0.5 * TreeSearchController.acc_high * (TreeSearchController.ACTION_STEP ** 2)
                    r_1 = r + dis_change
                    rr_1 = rr - TreeSearchController.acc_high * TreeSearchController.ACTION_STEP

                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0
                    elif rr_1 < 0:
                        self_v_2 = min(ego_info["speed"], TreeSearchController.v_high)
                        r_v_2 = min(r_veh["speed"] + TreeSearchController.acc_high, TreeSearchController.v_high)
                        dist_r = (r_v_2 ** 2 - TreeSearchController.v_low ** 2) / (2 * abs(TreeSearchController.acc_low))
                        dist_f = (self_v_2 ** 2 - TreeSearchController.v_low ** 2) / (2 * abs(TreeSearchController.acc_low)) + TreeSearchController.v_low * (-r_v_2 + self_v_2) / TreeSearchController.acc_low
                        r_2 = r_1 - dist_r + dist_f
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0

        if np.sum(pdf_array) == 0:
            return np.array([0, 1, 0])

        if self.vehicle.id == ego_info.get("veh_id"):
            return pdf_array
        else:
            return pdf_array / np.sum(pdf_array)

    @property
    def type(self):
        return self._type

    """
    Functions from NDDController
    """
    @property
    def ndd_pdf(self):
        current_time = self.env.get_simulation_time()
        if self._recent_ndd_pdf["time_step"] != current_time:
            veh_obs = self.vehicle_wrapper.observation.information
            cav_obs = self.env.ego_vehicle_wrapper.observation.information

            self._recent_ndd_pdf = self.get_ndd_pdf(obs=veh_obs, cav_obs=cav_obs)
            self._recent_ndd_pdf["time_step"] = current_time
        return self._recent_ndd_pdf["pdf"]

    def get_ndd_pdf(self, obs: object = None, cav_obs: object = None, external_use: object = False):
        _recent_ndd_pdf = {}
        longi_pdf, lateral_pdf, total_pdf = TreeSearchController.static_get_ndd_pdf(obs=obs, cav_obs=cav_obs)
        _recent_ndd_pdf["pdf"] = total_pdf
        if not external_use:
            _recent_ndd_pdf["time_step"] = self.env.world.get_snapshot().timestamp.elapsed_seconds
        return _recent_ndd_pdf

    @staticmethod
    # TODO: cav_obs???
    def static_get_ndd_pdf(obs: object = None, cav_obs: object = None):
        _, longi_pdf = TreeSearchController.Longitudinal_NDD(obs,cav_obs)
        _, _, lateral_pdf = TreeSearchController.Lateral_NDD(obs,cav_obs)
        total_pdf = [lateral_pdf[0], lateral_pdf[2]] + list(lateral_pdf[1] * longi_pdf)
        return longi_pdf, lateral_pdf, total_pdf

    @staticmethod
    def Longitudinal_NDD(obs: object = None, cav_obs: object = None):
        """
        Decide the Longitudinal acceleration
        Input: observation of surrounding vehicles
        Output: Acceleration
        """
        if not list(conf.CF_pdf_array):
            assert("No CF_pdf_array file!")
        if not list(conf.FF_pdf_array):
            assert("No FF_pdf_array file!")
        acc = 0
        ego_observation = obs["Ego"]
        v = ego_observation["speed"]
        f1 = obs["Lead"]
        if f1 is None:
            round_speed, round_speed_idx = TreeSearchController.round_to_(
                v, round_item="speed", round_to_closest=conf.v_resolution
            )
            pdf_array = conf.FF_pdf_array[round_speed_idx]
            if conf.safety_guard_enabled_flag:
                pdf_array = TreeSearchController._check_longitudinal_safety(obs, pdf_array)
            return acc, pdf_array

        else:
            r = f1["distance"]
            rr = f1["speed"] - v
            round_speed, round_speed_idx = TreeSearchController.round_to_(v, round_item="speed", round_to_closest=conf.v_resolution)
            round_r, round_r_idx = TreeSearchController.round_to_(r, round_item="range", round_to_closest=conf.r_resolution)
            round_rr, round_rr_idx = TreeSearchController.round_to_(rr, round_item="range_rate", round_to_closest=conf.rr_resolution)

            if (
                    not TreeSearchController._check_bound_constraints(r, conf.r_low, conf.r_high)
                    or not TreeSearchController._check_bound_constraints(rr, conf.rr_low, conf.rr_high)
                    or not TreeSearchController._check_bound_constraints(v, conf.v_low, conf.v_high)
            ):
                pdf_array = TreeSearchController.stochastic_IDM(ego_observation, f1)
                if conf.safety_guard_enabled_flag or conf.safety_guard_enabled_flag_IDM:
                    pdf_array = TreeSearchController._check_longitudinal_safety(obs, pdf_array)
                return acc, pdf_array

            pdf_array = conf.CF_pdf_array[round_r_idx, round_rr_idx, round_speed_idx]

            if sum(pdf_array) == 0:
                pdf_array = TreeSearchController.stochastic_IDM(ego_observation, f1)
                if conf.safety_guard_enabled_flag or conf.safety_guard_enabled_flag_IDM:
                    pdf_array = TreeSearchController._check_longitudinal_safety(obs, pdf_array)
                return acc, pdf_array

            if conf.safety_guard_enabled_flag:
                pdf_array = TreeSearchController._check_longitudinal_safety(obs, pdf_array)
            return acc, pdf_array

    @staticmethod
    def Lateral_NDD(obs: object = None, cav_obs: object = None):
        initial_pdf = np.array([0, 1, 0])
        try:
            if conf.OL_pdf is None or len(conf.OL_pdf) == 0:
                raise ValueError("No OL pdf file!")
        except ValueError as e:
            print(f"Caught an error: {e}")

        lane_id, v = obs["Ego"]["lane_index"], obs["Ego"]["speed"]
        f1, r1, f0, r0, f2, r2 = obs["Lead"], obs["Foll"], obs["LeftLead"], obs["LeftFoll"], obs["RightLead"], obs["RightFoll"]
        #print("f1 type:", type(f1), "value:", f1)

        if f1 is None:
            return False, 1, initial_pdf
        else:
            left_prob, still_prob, right_prob = 0, 0, 0
            LC_related_list = []
            LC_type_list = []

            for item in ["Left", "Right"]:
                if item == "Left":
                    surrounding = (f1, f0, r0)
                    left_prob, LC_type, LC_related = TreeSearchController._LC_prob(surrounding,obs)

                    LC_related_list.append(LC_related)
                    LC_type_list.append(LC_type)
                else:
                    surrounding = (f1, f2, r2)
                    right_prob,LC_type, LC_related = TreeSearchController._LC_prob(surrounding,obs)
                    LC_related_list.append(LC_related)
                    LC_type_list.append(LC_type)

            has_LC_data_on_at_least_one_side_flag = True

            if left_prob is None and right_prob is None:
                has_LC_data_on_at_least_one_side_flag = False

            if has_LC_data_on_at_least_one_side_flag:
                if left_prob is None:
                    left_prob = 0
                    right_prob = 2*right_prob
                elif right_prob is None:
                    right_prob = 0
                    left_prob = 2*left_prob

            has_CF_data_flag = TreeSearchController.check_whether_has_CF_data(obs["Ego"], f1)
            MOBIL_flag = ((not has_CF_data_flag) and (
                np.floor(v+0.5) <= 21)) or (not has_LC_data_on_at_least_one_side_flag)

            if MOBIL_flag:
                left_prob, right_prob = TreeSearchController.MOBIL_result(obs)
                LC_related_list =[(v), (v)]
                LC_type_list = ["MOBIL", "MOBIL"]

            if not obs["Ego"]["could_drive_adjacent_lane_left"]:
                left_prob = 0
                if conf.double_LC_prob_in_leftmost_rightest_flag:
                    right_prob *= 2
            if not obs["Ego"]["could_drive_adjacent_lane_right"]:
                right_prob = 0
                if conf.double_LC_prob_in_leftmost_rightest_flag:
                    left_prob *= 2

            if left_prob + right_prob > 1:
                tmp = left_prob + right_prob
                left_prob *= 0.9 / (tmp)
                right_prob *= 0.9 / (tmp)
            still_prob = 1 - left_prob - right_prob
            pdf_array = np.array([left_prob, still_prob, right_prob])

            lane_change_idx = np.random.choice( TreeSearchController.LANE_CHANGE_INDEX_LIST, None, False, pdf_array)
            if lane_change_idx != 1:
                return True, lane_change_idx, pdf_array
            else:
                return False, lane_change_idx, pdf_array

    @staticmethod
    def round_value_lane_change(real_value, value_list, round_item="speed"):
        if real_value < value_list[0]:
            real_value = value_list[0]
        elif real_value > value_list[-1]:
            real_value = value_list[-1]

        if conf.round_rule == "Round_to_closest":
            min_val, max_val, resolution = value_list[0], value_list[-1], value_list[1] - value_list[0]
            # real_value_old = np.clip(round((real_value - min_val) / resolution)*resolution + min_val, min_val, max_val)
            _num = (real_value-min_val)/resolution
            if int(_num*2) == _num*2:
                if int(_num) % 2 != 0:
                    _num += 0.5
            else:
                _num += 0.5
            real_value_new = int(_num)*resolution + min_val
            # assert real_value_new==real_value_old
            real_value = real_value_new

        if round_item == "speed":
            value_idx = bisect.bisect_left(value_list, real_value)
            value_idx = value_idx if real_value <= value_list[-1] else value_idx - 1
            try:
                assert value_idx <= (len(value_list)-1)
                assert value_idx >= 0
            except:
                print("Error in lane change round value")
            round_value = value_list[value_idx]
            return round_value, value_idx
        else:
            value_idx = bisect.bisect_left(value_list, real_value)
            value_idx = value_idx - \
                1 if real_value != value_list[value_idx] else value_idx
            try:
                assert value_idx <= (len(value_list)-1)
                assert value_idx >= 0
            except:
                print("Error in lane change round value")
            round_value = value_list[value_idx]
            return round_value, value_idx

    @staticmethod
    # @profile
    def _LC_prob(surrounding_vehicles, full_obs):
        """
        Input: (veh_front, veh_adj_front, veh_adj_back)
        output: the lane change probability and the expected lane change probability (take the ignored situation into account)
        """
        LC_prob, E_LC_prob = None, None
        veh_front, veh_adj_front, veh_adj_rear = surrounding_vehicles

        if not veh_adj_front and not veh_adj_rear:
            # One lead LC
            LC_prob, LC_related = TreeSearchController._get_One_lead_LC_prob(
                veh_front, full_obs)
            E_LC_prob = LC_prob
            return E_LC_prob, "One_lead", LC_related

        elif veh_adj_front and not veh_adj_rear:
            # Single lane change
            LC_prob, LC_related = TreeSearchController._get_Single_LC_prob(
                veh_front, veh_adj_front, full_obs)
            E_LC_prob = LC_prob
            return E_LC_prob, "SLC", LC_related

        elif not veh_adj_front and veh_adj_rear:
            # One Lead prob
            OL_LC_prob, OL_LC_related = TreeSearchController._get_One_lead_LC_prob(
                veh_front, full_obs)

            # Cut in prob
            CI_LC_prob, CI_LC_related = TreeSearchController._get_Cut_in_LC_prob(
                veh_front, veh_adj_rear, full_obs)
            LC_related = CI_LC_related

            r_adj = veh_adj_rear["distance"]

            if (r_adj >= conf.min_r_ignore) and (CI_LC_prob is not None) and (OL_LC_prob is not None):
                E_LC_prob = conf.ignore_adj_veh_prob * OL_LC_prob + \
                    (1-conf.ignore_adj_veh_prob) * CI_LC_prob
            else:
                E_LC_prob = CI_LC_prob
            return E_LC_prob, "Cut_in", LC_related

        elif veh_adj_front and veh_adj_rear:
            # Single lane change prob
            SLC_LC_prob, SLC_LC_related = TreeSearchController._get_Single_LC_prob(
                veh_front, veh_adj_front, full_obs)

            # Double lane change prob
            DLC_LC_prob, DLC_LC_related = TreeSearchController._get_Double_LC_prob(
                veh_adj_front, veh_adj_rear, full_obs)
            LC_related = DLC_LC_related

            r_adj = veh_adj_rear["distance"]

            if (r_adj >= conf.min_r_ignore) and (DLC_LC_prob is not None) and (SLC_LC_prob is not None):
                E_LC_prob = conf.ignore_adj_veh_prob * SLC_LC_prob + \
                    (1-conf.ignore_adj_veh_prob) * DLC_LC_prob
            else:
                E_LC_prob = DLC_LC_prob
            return E_LC_prob, "DLC", LC_related

    @staticmethod
    # @profile
    def _get_One_lead_LC_prob(veh_front, full_obs):
        v = full_obs["Ego"]["speed"]
        if not conf.enable_One_lead_LC:
            return 0, None
        r, rr = veh_front["distance"], veh_front["speed"] - v
        # Check bound
        if not TreeSearchController._check_bound_constraints(v, conf.one_lead_v_low,
                                                      conf.one_lead_v_high) or not TreeSearchController._check_bound_constraints(
                r, conf.one_lead_r_low, conf.one_lead_r_high) or not TreeSearchController._check_bound_constraints(rr,
                                                                                                            conf.one_lead_rr_low,
                                                                                                            conf.one_lead_rr_high):
            return 0, None

        round_r, round_r_idx = TreeSearchController.round_value_lane_change(
            real_value=r, value_list=conf.one_lead_r_list)
        round_rr, round_rr_idx = TreeSearchController.round_value_lane_change(
            real_value=rr, value_list=conf.one_lead_rr_list)
        round_speed, round_speed_idx = TreeSearchController.round_value_lane_change(
            real_value=v, value_list=conf.one_lead_speed_list, round_item="speed")
        # Since currently the OL raw data v>=24. So for v<=23, there is definitely no LC, so use the v==24 data when v<=23
        if round_speed <= 23 and conf.OL_LC_low_speed_flag:
            v_diff = conf.OL_LC_low_speed_use_v - round_speed
            assert (v_diff > 0)
            round_rr = round_rr - v_diff
            round_rr, round_rr_idx = TreeSearchController.round_value_lane_change(
                real_value=round_rr, value_list=conf.one_lead_rr_list)
            round_speed, round_speed_idx = TreeSearchController.round_value_lane_change(
                real_value=conf.OL_LC_low_speed_use_v, value_list=conf.one_lead_speed_list, round_item="speed")

        lane_change_prob = conf.OL_pdf[round_speed_idx, round_r_idx,
                           round_rr_idx, :][0] * conf.LANE_CHANGE_SCALING_FACTOR
        LC_related = (v, r, rr, round_speed, round_r, round_rr)

        # chech whether there is LC data in this case
        if conf.OL_pdf[round_speed_idx, round_r_idx, round_rr_idx, :][0] == 0 and \
                conf.OL_pdf[round_speed_idx, round_r_idx, round_rr_idx, :][1] == 0:
            lane_change_prob = None

        return lane_change_prob, LC_related

    @staticmethod
    # @profile
    def _get_Double_LC_prob(veh_adj_front, veh_adj_rear, full_obs):
        v = full_obs["Ego"]["speed"]
        v_list, r1_list, r2_list, rr1_list, rr2_list = conf.lc_v_list, conf.lc_rf_list, conf.lc_re_list, conf.lc_rrf_list, conf.lc_rre_list
        LC_related = None
        # Double lane change
        if not conf.enable_Double_LC:
            return 0, LC_related
        r1, rr1 = veh_adj_front["distance"], veh_adj_front["speed"] - v
        r2, rr2 = veh_adj_rear["distance"], v - veh_adj_rear["speed"]
        if not TreeSearchController._check_bound_constraints(v, conf.lc_v_low, conf.lc_v_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(r1, conf.lc_rf_low, conf.lc_rf_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(rr1, conf.lc_rrf_low, conf.lc_rrf_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(r2, conf.lc_re_low, conf.lc_re_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(rr2, conf.lc_rre_low, conf.lc_rre_high):
            return 0, LC_related
        round_v, v_idx = TreeSearchController.round_value_lane_change(
            real_value=v, value_list=v_list, round_item="speed")
        round_r1, r1_idx = TreeSearchController.round_value_lane_change(
            real_value=r1, value_list=r1_list)
        round_rr1, rr1_idx = TreeSearchController.round_value_lane_change(
            real_value=rr1, value_list=rr1_list)
        round_r2, r2_idx = TreeSearchController.round_value_lane_change(
            real_value=r2, value_list=r2_list)
        round_rr2, rr2_idx = TreeSearchController.round_value_lane_change(
            real_value=rr2, value_list=rr2_list)

        lane_change_prob = conf.DLC_pdf[v_idx, r1_idx, rr1_idx,
                           r2_idx, rr2_idx, :][0] * conf.LANE_CHANGE_SCALING_FACTOR

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1,
                      round_rr1, round_r2, round_rr2, lane_change_prob)

        # chech whether there is LC data in this case
        if conf.DLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0] == 0 and \
                conf.DLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][1] == 0:
            lane_change_prob = None

        return lane_change_prob, LC_related

    @staticmethod
    # @profile
    def _get_Single_LC_prob(veh_front, veh_adj_front, full_obs):
        v = full_obs["Ego"]["speed"]
        v_list, r1_list, r2_list, rr1_list, rr2_list = conf.lc_v_list, conf.lc_rf_list, conf.lc_re_list, conf.lc_rrf_list, conf.lc_rre_list
        LC_related = None
        # Single lane change
        if not conf.enable_Single_LC:
            return 0, LC_related

        r1, rr1 = veh_front["distance"], veh_front["speed"] - v
        r2, rr2 = veh_adj_front["distance"], veh_adj_front["speed"] - v

        if not TreeSearchController._check_bound_constraints(v, conf.lc_v_low, conf.lc_v_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(r1, conf.lc_rf_low, conf.lc_rf_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(rr1, conf.lc_rrf_low, conf.lc_rrf_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(r2, conf.lc_rf_low, conf.lc_rf_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(rr2, conf.lc_rrf_low, conf.lc_rrf_high):
            return 0, LC_related

        round_v, v_idx = TreeSearchController.round_value_lane_change(
            real_value=v, value_list=v_list, round_item="speed")
        round_r1, r1_idx = TreeSearchController.round_value_lane_change(
            real_value=r1, value_list=r1_list)
        round_rr1, rr1_idx = TreeSearchController.round_value_lane_change(
            real_value=rr1, value_list=rr1_list)
        round_r2, r2_idx = TreeSearchController.round_value_lane_change(
            real_value=r2, value_list=r2_list)
        round_rr2, rr2_idx = TreeSearchController.round_value_lane_change(
            real_value=rr2, value_list=rr2_list)

        lane_change_prob = conf.SLC_pdf[v_idx, r1_idx, rr1_idx,
                           r2_idx, rr2_idx, :][0] * conf.LANE_CHANGE_SCALING_FACTOR

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1,
                      round_rr1, round_r2, round_rr2, lane_change_prob)

        # check whether there is LC data in this case
        if conf.SLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0] == 0 and \
                conf.SLC_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][1] == 0:
            lane_change_prob = None

        return lane_change_prob, LC_related

    @staticmethod
    # @profile
    def _get_Cut_in_LC_prob(veh_front, veh_adj_rear, full_obs):
        v = full_obs["Ego"]["speed"]
        v_list, r1_list, r2_list, rr1_list, rr2_list = conf.lc_v_list, conf.lc_rf_list, conf.lc_re_list, conf.lc_rrf_list, conf.lc_rre_list
        LC_related = None

        if not conf.enable_Cut_in_LC:
            return 0, None

        r1, rr1 = veh_front["distance"], veh_front["speed"] - v
        r2, rr2 = veh_adj_rear["distance"], v - veh_adj_rear["speed"]

        if not TreeSearchController._check_bound_constraints(v, conf.lc_v_low, conf.lc_v_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(r1, conf.lc_rf_low, conf.lc_rf_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(rr1, conf.lc_rrf_low, conf.lc_rrf_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(r2, conf.lc_rf_low, conf.lc_rf_high):
            return 0, LC_related
        elif not TreeSearchController._check_bound_constraints(rr2, conf.lc_rrf_low, conf.lc_rrf_high):
            return 0, LC_related

        round_v, v_idx = TreeSearchController.round_value_lane_change(
            real_value=v, value_list=v_list, round_item="speed")
        round_r1, r1_idx = TreeSearchController.round_value_lane_change(
            real_value=r1, value_list=r1_list)
        round_rr1, rr1_idx = TreeSearchController.round_value_lane_change(
            real_value=rr1, value_list=rr1_list)
        round_r2, r2_idx = TreeSearchController.round_value_lane_change(
            real_value=r2, value_list=r2_list)
        round_rr2, rr2_idx = TreeSearchController.round_value_lane_change(
            real_value=rr2, value_list=rr2_list)

        lane_change_prob = conf.CI_pdf[v_idx, r1_idx, rr1_idx,
                           r2_idx, rr2_idx, :][0] * conf.LANE_CHANGE_SCALING_FACTOR

        LC_related = (v, r1, rr1, r2, rr2, round_v, round_r1,
                      round_rr1, round_r2, round_rr2, lane_change_prob)

        # chech whether there is LC data in this case
        if conf.CI_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][0] == 0 and \
                conf.CI_pdf[v_idx, r1_idx, rr1_idx, r2_idx, rr2_idx, :][1] == 0:
            lane_change_prob = None

        return lane_change_prob, LC_related

    @staticmethod
    def round_to_(val, round_item, round_to_closest):
        """
        round the val to the round_to_closest (for example 1.0, 0.2 ...)
        """
        if round_item == "speed":
            value_list = conf.speed_list
        elif round_item == "range":
            value_list = conf.r_list
        elif round_item == "range_rate":
            value_list = conf.rr_list

        if round_to_closest == 1:
            mul, add, check = 1, 1, 0.5
        elif round_to_closest == 0.5:
            mul, add, check = 2, 0.5, 0.25
        elif round_to_closest == 0.2:
            mul, add, check = 5, 0.2, 0.1

        if val < value_list[0]:
            val = value_list[0]
        elif val > value_list[-1]:
            val = value_list[-1]

        if conf.round_rule == "Round_to_closest":
            round_val = np.floor(val * mul + 0.5) / mul
            try:
                assert (-check < round_val - val <= check + 1e-10)
            except:
                round_val += add
                try:
                    assert (-check < round_val - val <= check + 1e-10)
                except:
                    print(val, round_val)
                    raise ValueError("Round error!")

        try:
            round_idx = value_list.index(round_val)
        except:
            round_idx = min(range(len(value_list)),
                            key=lambda i: abs(value_list[i] - round_val))
            assert (np.abs(value_list[round_idx] - round_val) < 1e-8)

        return round_val, round_idx

    @staticmethod
    def _check_bound_constraints(value, bound_low, bound_high):
        if value < bound_low or value > bound_high:
            return False
        else:
            return True


    @staticmethod
    def stochastic_IDM(ego_vehicle, front_vehicle):
        tmp_acc = utils.acceleration(
            ego_vehicle=ego_vehicle, front_vehicle=front_vehicle) #TODO: should be compatible to carla, double check
        tmp_acc = np.clip(tmp_acc, conf.acc_low, conf.acc_high)
        acc_possi_list = scipy.stats.norm.pdf(conf.acc_list, tmp_acc, 0.3)
        # clip the possibility to avoid too small possibility
        acc_possi_list = [
            val if val > conf.Stochastic_IDM_threshold else 0 for val in acc_possi_list]
        assert(sum(acc_possi_list) > 0)
        acc_possi_list = acc_possi_list/(sum(acc_possi_list))
        return acc_possi_list

    @staticmethod
    # @profile
    def MOBIL_result(obs):
        """
        Given that now is using the MOBIL model, calculate the left/ right turn probability suggested by the MOBIL model
        """
        left_prob, right_prob = 0, 0
        if not conf.enable_MOBIL:
            return left_prob, right_prob
        lane_id = obs["Ego"]["lane_index"]

        MOBIL_LC_prob = 1e-2
        left_gain, right_gain = -np.inf, -np.inf
        left_LC_flag, right_LC_flag = False, False
        for lc_decision in ["left", "right"]:
            LC_flag, gain = TreeSearchController._MOBIL_model(lc_decision, obs)
            if LC_flag:
                if lc_decision == "right":
                    right_LC_flag, right_gain = LC_flag, gain
                elif lc_decision == "left":
                    left_LC_flag, left_gain = LC_flag, gain

        if left_LC_flag or right_LC_flag:
            if left_gain >= right_gain:
                left_prob, right_prob = MOBIL_LC_prob, 0.
            else:
                left_prob, right_prob = 0., MOBIL_LC_prob
            # assert(left_prob+right_prob == 1)
        return left_prob, right_prob


    @staticmethod
    # @profile
    def _MOBIL_model(lc_decision, obs):
        """
        Mobil  model for the NDD vehicle Lane change
        :param lane_index: the candidate lane for the change
        :return:
            The flag for whether do lane change. The gain for this lane change maneuver.
            The first output stands for lane change flag. The False could be crash immediately after doing LC or the gain is smaller than the required MOBIL Model parameters.
            The second output is gain when it decides to do the LC.
        """
        gain = None
        if lc_decision == "left":
            new_preceding, new_following = obs["LeftLead"], obs["LeftFoll"]
        elif lc_decision == "right":
            new_preceding, new_following = obs["RightLead"], obs["RightFoll"]

        # Check whether will crash immediately
        r_new_preceding, r_new_following = 99999, 99999
        if new_preceding:
            r_new_preceding = new_preceding["distance"]
        if new_following:
            r_new_following = new_following["distance"]
        if r_new_preceding <= 0 or r_new_following <= 0:
            return False, gain

        new_following_a = utils.acceleration(
            ego_vehicle=new_following, front_vehicle=new_preceding)
        new_following_pred_a = utils.acceleration(
            ego_vehicle=new_following, front_vehicle=obs["Ego"])

        # The deceleration of the new following vehicle after the the LC should not be too big (negative)
        if new_following_pred_a < -conf.NDD_LANE_CHANGE_MAX_BRAKING_IMPOSED:
            return False, gain

        old_preceding, old_following = obs["Lead"], obs["Foll"]
        self_pred_a = utils.acceleration(
            ego_vehicle=obs["Ego"], front_vehicle=new_preceding)

        # Is there an acceleration advantage for ego and/or ego vheicle's followers to change lane?
        self_a = utils.acceleration(
            ego_vehicle=obs["Ego"], front_vehicle=old_preceding)
        old_following_a = utils.acceleration(
            ego_vehicle=old_following, front_vehicle=obs["Ego"])
        old_following_pred_a = utils.acceleration(
            ego_vehicle=old_following, front_vehicle=old_preceding)
        gain = self_pred_a - self_a + conf.NDD_POLITENESS * \
            (new_following_pred_a - new_following_a +
             old_following_pred_a - old_following_a)
        if gain <= conf.NDD_LANE_CHANGE_MIN_ACC_GAIN:
            return False, gain
        return True, gain

    @staticmethod
    def check_whether_has_CF_data(ego, f1):
        """
        If there is no CF data, then use IDM+MOBIL
        """
        v = ego["speed"]
        r = f1["distance"]
        rr = f1["speed"] - v
        round_speed, round_speed_idx = TreeSearchController.round_to_(
            v, round_item="speed", round_to_closest=conf.v_resolution)
        round_r, round_r_idx = TreeSearchController.round_to_(
            r, round_item="range", round_to_closest=conf.r_resolution)
        round_rr, round_rr_idx = TreeSearchController.round_to_(
            rr, round_item="range_rate", round_to_closest=conf.rr_resolution)

        pdf_array = conf.CF_pdf_array[round_r_idx,
                                      round_rr_idx, round_speed_idx]
        if sum(pdf_array) == 0:
            return False
        else:
            return True

    """
    Functions from NADEController
    """

    def get_NDD_possi(self):
        if hasattr(self, 'ndd_pdf'):
            print(f"[Debug] get_NDD_possi() returning self.ndd_pdf = {self.ndd_pdf}")
            return self.ndd_pdf
        else:
            print(f"[Debug] get_NDD_possi() called but ndd_pdf not set")
            return [0.0, 1.0, 0.0]  # fallback

    def _sample_critical_action(self, bv_criticality, criticality_array, ndd_possi_array, epsilon=conf.epsilon_value):
        normalized_critical_pdf_array = criticality_array / bv_criticality
        epsilon_pdf_array = utils.epsilon_greedy(normalized_critical_pdf_array, ndd_possi_array, epsilon=epsilon)
        bv_action_idx = np.random.choice(len(conf.BV_ACTIONS), 1, replace=False, p=epsilon_pdf_array)
        critical_possi, ndd_possi = epsilon_pdf_array[bv_action_idx], ndd_possi_array[bv_action_idx]
        weight_list = (ndd_possi_array+1e-30)/(epsilon_pdf_array+1e-30)
        weight = ndd_possi/critical_possi
        self.bv_criticality_array = criticality_array
        self.normalized_critical_pdf_array = normalized_critical_pdf_array
        self.ndd_possi_array = ndd_possi_array
        self.epsilon_pdf_array = epsilon_pdf_array
        self.weight = weight.item()
        self.ndd_possi = ndd_possi.item()
        self.critical_possi = critical_possi.item()
        return bv_action_idx, weight, ndd_possi, critical_possi, weight_list

    @staticmethod
    def _hard_brake_challenge(v, r, rr):
        """Calculate the hard brake challenge value.
           Situation: BV in front of the CAV, do the hard-braking.

        Args:
            v (float): Speed of BV.
            r (float): Distance between BV and CAV.
            rr (float): Range rate of BV and CAV.

        Returns:
            list(float): List of challenge for the BV behavior.
        """
        CF_challenge_array = np.zeros((len(conf.BV_ACTIONS)-2), dtype=float)
        round_speed, round_r, round_rr = utils._round_data_plain(v, r, rr)
        index = np.where(
            (conf.CF_state_value == [round_r, round_rr, round_speed]).all(1))
        assert(len(index) <= 1)
        index = index[0]
        if len(index):
            CF_challenge_array = conf.CF_challenge_value[index.item(), :]
        new_r = r + rr
        if new_r <= 2.1:
            CF_challenge_array = np.ones(
            (len(conf.BV_ACTIONS) - 2), dtype=float)
        return CF_challenge_array

    @staticmethod
    def _BV_accelerate_challenge(v, r, rr):
        """Assume the CAV is cutting in the BV and calculate by the BV CF

        Args:
            v (float): Speed of BV.
            r (float): Distance between BV and CAV.
            rr (float): Range rate between BV and CAV.

        Returns:
            float: Challenge of the BV behavior.
        """
        BV_CF_challenge_array = np.zeros(
            (len(conf.BV_ACTIONS) - 2), dtype=float)
        round_speed, round_r, round_rr = utils._round_data_plain(v, r, rr)

        index = np.where((conf.BV_CF_state_value == [
                         round_r, round_rr, round_speed]).all(1))
        assert(len(index) <= 1)
        index = index[0] if len(index) > 0 else []

        if len(index):
            BV_CF_challenge_array = conf.BV_CF_challenge_value[index.item(), :]
        new_r = r + rr
        if new_r <= 2.1:
            BV_CF_challenge_array = np.ones(
            (len(conf.BV_ACTIONS) - 2), dtype=float)
        return BV_CF_challenge_array

    def Decompose_decision(self, CAV, SM_LC_prob, full_obs=None, predicted_full_obs=None, predicted_traj_obs=None):
        self.bv_criticality_array = np.zeros(len(conf.ACTIONS), dtype=float)
        bv_criticality, bv_action_idx, weight, ndd_possi, critical_possi, weight_list, criticality_array = - \
            np.inf, None, None, None, None, None, np.zeros(len(conf.ACTIONS), dtype=float)
        bv_id = self.vehicle.id
        bv_pdf = self.get_NDD_possi()
        bv_obs = self.env.step_data[bv_id]["observation"]
        bv_left_prob, bv_right_prob = bv_pdf[0], bv_pdf[1]

        if not ((0.99999 <= bv_left_prob <= 1) or (0.99999 <= bv_right_prob <= 1)):
            bv_criticality, criticality_array, bv_challenge_array, risk = self._calculate_criticality(
                bv_obs, CAV, SM_LC_prob, full_obs, predicted_full_obs, predicted_traj_obs)
            self.bv_challenge_array = bv_challenge_array
        return bv_criticality, criticality_array


    def Decompose_sample_action(self, bv_criticality, bv_criticality_array, bv_pdf, epsilon=conf.epsilon_value):
        """ Sample the critical action of the BV.
        """
        if epsilon is None:
            epsilon = conf.epsilon_value
        bv_action_idx, weight, ndd_possi, critical_possi, weight_list = None, None, None, None, None
        if bv_criticality > conf.criticality_threshold:
            bv_action_idx, weight, ndd_possi, critical_possi, weight_list = self._sample_critical_action(
                bv_criticality, bv_criticality_array, bv_pdf, epsilon)
        if weight is not None:
            weight, ndd_possi, critical_possi = weight.item(
            ), ndd_possi.item(), critical_possi.item()
        return bv_action_idx, weight, ndd_possi, critical_possi, weight_list
    """
    Functions from Tree search Controller
    """
    def step(self):
        self.ego_info = self.vehicle_wrapper.observation.information["Ego"]
        final_pdf = self.ndd_pdf
        if self.vehicle_wrapper.controlled_duration == 0:
            action_id = np.random.choice(
                len(conf.BV_ACTIONS), 1, replace=False, p=final_pdf).item()
            self.action = utils.action_id_to_action_command(action_id)
        return final_pdf

    @staticmethod
    def update_single_vehicle_obs_no_action(veh, duration = 1.0, is_lane_change = None, action= None):
        initial_traj = Traj(veh["position"][0], veh["position"][1], veh["speed"], veh["lateral_speed"], veh["lane_index"])
        if is_lane_change is None:
            is_lane_change = utils.is_lane_change(veh) # TODO: double check whether this is correct in CARLA env
        new_traj_result = initial_traj.predict_without_action(initial_traj, duration, is_lane_change, action)
        new_veh = dict(veh)
        new_veh["position"] = (new_traj_result["x_lon"],
                               new_traj_result["x_lat"])
        new_veh["speed"] = new_traj_result["v_lon"]
        new_veh["lateral_velocity"] = new_traj_result["v_lat"]
        new_veh["lane_index"] = new_traj_result["lane_index"]
        return new_veh, initial_traj

    @staticmethod
    def is_CF(cav_obs, bv_obs):
        CF_info= False
        bv_r1 = bv_obs.get("Foll")
        bv_f1 = bv_obs.get("Lead")
        CAV_id = cav_obs["Ego"]["veh_id"] # TODO: double check whether it's the correct structure

        bv_v, bv_range_CAV, bv_rangerate_CAV = None, None, None
        if bv_r1 and bv_r1.get("veh_id") == CAV_id:
            # CAV is following BV
            CF_info = "CAV_BV"
            bv_v = bv_obs["Ego"]["speed"]
            bv_range_CAV = bv_obs["Ego"]["position"][0] - cav_obs["Ego"]["position"][0] - conf.LENGTH
            bv_rangerate_CAV = bv_obs["Ego"]["speed"] - cav_obs["Ego"]["speed"]

        elif bv_f1 and bv_f1.get("veh_id") == CAV_id:
            # BV is following CAV
            CF_info = "BV_CAV"
            bv_v = bv_obs["Ego"]["speed"]
            bv_range_CAV = cav_obs["Ego"]["position"][0] - bv_obs["Ego"]["position"][0] - conf.LENGTH
            bv_rangerate_CAV = cav_obs["Ego"]["speed"] - bv_obs["Ego"]["speed"]

        return CF_info, bv_v, bv_range_CAV, bv_rangerate_CAV

    @staticmethod
    def update_single_vehicle_obs(veh, action, duration=conf.simulation_resolution):
        new_veh = dict(veh) #TODO

        if (action == "left" or action == "right") and not utils.is_lane_change(veh):
            traj = Traj(veh["position"][0], veh["position"][1],
                        veh["speed"], veh["lateral_velocity"], veh["lane_index"],
                        carla_map=conf.carla_map)  # TODO

            new_traj_result = traj.predict_with_action(action, 0.0, duration)
            traj.crop(0.0, duration)

            new_veh["position"] = (new_traj_result["x_lon"], new_traj_result["x_lat"])
            new_veh["speed"] = new_traj_result["v_lon"]
            new_veh["lateral_velocity"] = new_traj_result["v_lat"]
            new_veh["road_id"] = new_traj_result["road_id"]
            new_veh["lane_index"] = new_traj_result["lane_index"]
            new_veh["could_drive_adjacent_lane_left"] = True
            new_veh["could_drive_adjacent_lane_right"] = True
        else:
            new_veh, traj = TreeSearchController.update_single_vehicle_obs_no_action(veh, duration=duration, is_lane_change=None, action=action)

        return new_veh, traj

    @staticmethod
    # @profile
    def traj_to_obs(prev_full_obs, full_traj, time):
        obs = {}
        for key in prev_full_obs:
            obs[key] = dict(prev_full_obs[key])
        for veh_id in full_traj:
            obs[veh_id]["position"] = (
                full_traj[veh_id].traj_info[time]["x_lon"], full_traj[veh_id].traj_info[time]["x_lat"])
            obs[veh_id]["speed"] = full_traj[veh_id].traj_info[time]["v_lon"]
            obs[veh_id]["lateral_velocity"] = full_traj[veh_id].traj_info[time]["v_lat"]
            obs[veh_id]["lane_index"] = full_traj[veh_id].traj_info[time]["lane_index"]
        return obs

    @staticmethod
    # @profile
    def update_obs(full_obs, cav_id, bv_id, bv_action, cav_action, predicted_full_obs=None, predicted_full_traj=None):
        new_full_obs = {}
        for key in full_obs:
            new_full_obs[key] = dict(full_obs[key])
        trajectory_obs = {}
        for veh_id in full_obs:
            action = "still"
            if veh_id == bv_id:
                action = bv_action
            if veh_id == cav_id:
                continue
            if action:
                vehicle = new_full_obs[veh_id]
                if predicted_full_obs is None or predicted_full_traj is None:
                    new_full_obs[veh_id], trajectory_obs[
                        veh_id] = TreeSearchController.update_single_vehicle_obs(
                        vehicle, action)
                else:
                    new_full_obs[veh_id] = predicted_full_obs[veh_id][action]
                    trajectory_obs[veh_id] = predicted_full_traj[veh_id][action]
        new_full_obs[cav_id], trajectory_obs[cav_id] = TreeSearchController.update_single_vehicle_obs(
            new_full_obs[cav_id], cav_action)

        # Sort the observation using the distance from the CAV
        av_pos = new_full_obs["CAV"]["position"]
        for veh_id in new_full_obs:
            bv_pos = new_full_obs[veh_id]["position"]
            new_full_obs[veh_id]["euler_distance"] = utils.cal_euclidean_dist(
                av_pos, bv_pos)
        new_full_obs = OrderedDict(
            sorted(new_full_obs.items(), key=lambda item: item[1]['euler_distance']))
        for traj in trajectory_obs:
            trajectory_obs[traj].crop(0.0, 1.0)
        return new_full_obs, trajectory_obs

    @staticmethod
    # @profile
    def cav_bv_obs_to_full_obs(cav_obs, bv_obs):
        """get full observation from cav and bv observation"""
        full_obs = {}
        for cav_observe_info in cav_obs:
            if cav_obs[cav_observe_info] is None:
                continue
            observed_id = cav_obs[cav_observe_info]["veh_id"]
            if observed_id not in full_obs:
                full_obs[observed_id] = cav_obs[cav_observe_info]
        for bv_observe_info in bv_obs:
            if bv_obs[bv_observe_info] is None:
                continue
            observed_id = bv_obs[bv_observe_info]["veh_id"]
            if observed_id not in full_obs:
                full_obs[observed_id] = bv_obs[bv_observe_info]
        return full_obs

    @staticmethod
    # @profile
    def full_obs_to_cav_bv_obs(full_obs, cav_id, bv_id):
        new_full_obs = {}
        for key in full_obs:
            new_full_obs[key] = dict(full_obs[key])
        cav_obs = TreeSearchController.full_obs_to_single_obs(
            new_full_obs, cav_id)
        bv_obs = TreeSearchController.full_obs_to_single_obs(
            new_full_obs, bv_id)
        return cav_obs, bv_obs

    @staticmethod
    # @profile
    def _process_info(full_obs, ego_id=None, longi=1, lateral=0):
        ego_length = 5
        ego_lane_index = full_obs[ego_id]["lane_index"]
        ego_lane_pos = full_obs[ego_id]["position"][0]
        cand_id = None
        cand_dist = 0
        for bv_id in full_obs:
            if bv_id != ego_id:
                bv_length = 5
                bv_lane_index = full_obs[bv_id]["lane_index"]
                bv_lane_pos = full_obs[bv_id]["position"][0]
                if bv_lane_index == ego_lane_index + lateral and longi * (bv_lane_pos - ego_lane_pos) >= 0:
                    dist = abs(bv_lane_pos - ego_lane_pos)
                    if longi == 1:
                        dist -= ego_length
                    if longi == -1:
                        dist -= bv_length
                    if not cand_id:
                        cand_id = bv_id
                        cand_dist = dist
                    elif cand_dist > dist:
                        cand_id = bv_id
                        cand_dist = dist
        if cand_id is None:
            veh = None
        else:
            veh = full_obs[cand_id]
            veh["distance"] = cand_dist
        return veh

    @staticmethod
    # @profile
    def full_obs_to_single_obs(full_obs, veh_id):
        obs = {"Ego": full_obs[veh_id]}
        obs["Lead"] = TreeSearchController._process_info(
            full_obs, veh_id, longi=1, lateral=0)
        obs["LeftLead"] = TreeSearchController._process_info(
            full_obs, veh_id, longi=1, lateral=1)
        obs["RightLead"] = TreeSearchController._process_info(
            full_obs, veh_id, longi=1, lateral=-1)
        obs["Foll"] = TreeSearchController._process_info(
            full_obs, veh_id, longi=-1, lateral=0)
        obs["LeftFoll"] = TreeSearchController._process_info(
            full_obs, veh_id, longi=-1, lateral=1)
        obs["RightFoll"] = TreeSearchController._process_info(
            full_obs, veh_id, longi=-1, lateral=-1)
        return obs

    @staticmethod
    # @profile
    def crash_check(cav_obs, bv_obs, cav_id, bv_id, previous_obs, traj):
        if traj is None:
            return False
        cav_traj = traj[cav_id]
        bv_traj = traj[bv_id]
        return collision_check(cav_traj.traj_info, bv_traj.traj_info)

    @staticmethod
    # @profile
    def leaf_node_check(full_obs, previous_obs, traj, cav_obs, bv_obs, cav_id, bv_id, search_depth):
        """check whether the search can be terminated

        Args:
            CAV (vehicle.observation.information["Ego"]): CAV observation information
            BV (vehicle.observation.information["Ego"]): the vehicle observation
            all_candidates (observation list): all NADE candidates for NADE decision
            CAV_action (int): the action of CAV
            BV_action (int): the action of BV
            search_depth (int): current search depth

        Returns:
            [type]: [description]
        """
        challenge_array = np.zeros(
            TreeSearchController.ACTION_NUM)
        depth_flag = (search_depth ==
                      TreeSearchController.MAX_TREE_SEARCH_DEPTH)
        CF_flag = TreeSearchController.is_CF(cav_obs, bv_obs)[0]
        bv_id = bv_obs["Ego"]["veh_id"]
        if CF_flag:
            challenge_array[2:] = TreeSearchController.get_CF_challenge_array(
                cav_obs, bv_obs)
        crash_flag = TreeSearchController.crash_check(
            cav_obs, bv_obs, cav_id, bv_id, previous_obs, traj)
        if crash_flag:
            challenge_array = np.ones(
                TreeSearchController.ACTION_NUM)
        return depth_flag or crash_flag, challenge_array

    @staticmethod
    # @profile
    def tree_search_maneuver_challenge(full_obs, previous_obs, traj, cav_id, bv_id, search_depth, cav_obs=None,
                                       bv_obs=None, predicted_full_obs=None, predicted_full_traj=None):
        """generate the maneuver challenge value for a CAV and BV pair.

        Args:
            full_obs: all the controlled BV candidates
            cav_id (int): the CAV id
            bv_id (int): the BV id that will be controlled
            search_depth (int): the depth of the tree

        Returns:
            float: challenge(the prob of having crashes)
        """
        challenge_array = np.zeros(
            TreeSearchController.ACTION_NUM)
        if (not cav_obs) or (not bv_obs):
            cav_obs, bv_obs = TreeSearchController.full_obs_to_cav_bv_obs(
                full_obs, cav_id, bv_id)
        leaf_flag, leaf_challenge_array = TreeSearchController.leaf_node_check(
            full_obs, previous_obs, traj, cav_obs, bv_obs, cav_id, bv_id, search_depth)
        cav_action_dict, bv_action_dict, cav_pdf, bv_pdf = TreeSearchController.get_cav_bv_pdf(
            cav_obs, bv_obs)
        # cav_action_list = ["brake", "still", "accelerate"]
        cav_action_list = ["still"]
        if leaf_flag:
            max_challenge_at_leaf = np.max(leaf_challenge_array)
            new_leaf_challenge_array = max_challenge_at_leaf * np.ones_like(leaf_challenge_array)
            return new_leaf_challenge_array, bv_pdf
        else:
            # estimate the maneuver challenge for each bv maneuver
            for bv_action in bv_action_dict:
                if bv_action_dict[bv_action] == 0:
                    continue
                else:
                    challenge_tmp_list = []
                    for cav_action in cav_action_list:
                        updated_full_obs, trajectory_obs = TreeSearchController.update_obs(
                            full_obs, cav_id, bv_id, bv_action, cav_action, predicted_full_obs, predicted_full_traj)
                        new_challenge_array, updated_bv_pdf = TreeSearchController.tree_search_maneuver_challenge(
                            updated_full_obs, full_obs, trajectory_obs, cav_id, bv_id, search_depth + 1)
                        discount_factor = 1
                        if search_depth != 0:
                            discount_factor = conf.treesearch_config["treesearch_discount_factor"]
                        challenge_tmp = discount_factor * np.sum(new_challenge_array * updated_bv_pdf)
                        challenge_tmp_list.append(challenge_tmp)
                    challenge_array[TreeSearchController.ACTION_TYPE[bv_action]] += max(
                        challenge_tmp_list)
        return challenge_array, bv_pdf

    @staticmethod
    # @profile
    def get_CF_challenge_array(cav_obs, bv_obs):
        CF_info, bv_v, bv_range_CAV, bv_rangerate_CAV = TreeSearchController.is_CF(
            cav_obs, bv_obs)
        if not CF_info:
            raise ValueError("get CF challenge in non-CF mode")
        if CF_info == "CAV_BV":
            return TreeSearchController._hard_brake_challenge(bv_v, bv_range_CAV, bv_rangerate_CAV)
        elif CF_info == "BV_CAV":
            return TreeSearchController._BV_accelerate_challenge(bv_v, bv_range_CAV, bv_rangerate_CAV)

    @staticmethod
    # @profile
    def get_cav_bv_pdf(cav_obs, bv_obs):
        cav_pdf = TreeSearchController.SURROGATE_MODEL_FUNCTION(
            cav_obs)
        _, _, bv_pdf = TreeSearchController.static_get_ndd_pdf(bv_obs)
        if utils.is_lane_change(cav_obs["Ego"]):
            cav_action_dict = {"left": 0, "right": 0, "still": 1}
            cav_pdf = [0, 1, 0]
        else:
            cav_action_dict = {
                "left": cav_pdf[0], "right": cav_pdf[2], "still": cav_pdf[1]}
        if utils.is_lane_change(bv_obs["Ego"]):
            bv_action_dict = {"left": 0, "right": 0, "still": 1}
            bv_pdf[0] = 0
            bv_pdf[1] = 0
            bv_pdf[2:] = 1.0 / (len(bv_pdf[2:])) * np.ones_like(bv_pdf[2:])
            bv_pdf = bv_pdf / np.sum(bv_pdf)
        else:
            bv_action_dict = {
                "left": bv_pdf[0], "right": bv_pdf[1], "still": np.sum(bv_pdf[2:])}
        return cav_action_dict, bv_action_dict, cav_pdf, bv_pdf

    @staticmethod
    # @profile
    def _calculate_criticality(bv_obs, CAV, SM_LC_prob, full_obs=None, predicted_full_obs=None,
                               predicted_full_traj=None):
        """calculate the criticality of the BV: Feng, S., Yan, X., Sun, H., Feng, Y. and Liu, H.X., 2021. Intelligent driving intelligence test for autonomous vehicles with naturalistic and adversarial environment. Nature communications, 12(1), pp.1-14.


        Args:
            CAV (vehicle): the CAV in the environment
            SM_LC_prob (list): the left, still and right turn probabiltiy

        Returns:
            array: criticality array of a specific CAV
        """
        _, _, bv_pdf = TreeSearchController.static_get_ndd_pdf(bv_obs)
        if full_obs is None:
            full_obs = TreeSearchController.cav_bv_obs_to_full_obs(
                bv_obs, CAV)

        # Maneuver challenge calculation
        if utils.is_lane_change(bv_obs["Ego"]) and conf.experiment_config["mode"] != "risk_NDE":
            # if the vehicle is doing lane change, then do not need to calculate the maneuver challenge
            bv_challenge_array, updated_bv_pdf = np.zeros(TreeSearchController.ACTION_NUM), np.zeros(
                TreeSearchController.ACTION_NUM)
        else:
            bv_challenge_array, updated_bv_pdf = TreeSearchController.tree_search_maneuver_challenge(
                full_obs, None, None, CAV["Ego"]["veh_id"], bv_obs["Ego"]["veh_id"], 0, CAV, bv_obs, predicted_full_obs,
                predicted_full_traj)

        # Criticality calculation: maneuver challenge * exposure frequency
        bv_criticality_array = bv_pdf * bv_challenge_array
        risk = np.sum(updated_bv_pdf * bv_challenge_array)
        bv_criticality = np.sum(bv_criticality_array)
        return bv_criticality, bv_criticality_array, bv_challenge_array, risk