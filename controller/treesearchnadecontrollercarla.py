import math
from os import times
from numpy.core.numeric import full
from mtlsp import utils
from controller.nadecontrollercarla import NADEBackgroundController
import numpy as np
from controller.nddcontroller import NDDController
import conf.conf as conf
import utils
from math import isclose
from mtlsp.controller.vehicle_controller.idmcontroller import IDMController
from collections import OrderedDict
from controller.traj_predictor import Traj

veh_length = 5.0
veh_width = 2.0
circle_r = 1.3
tem_len = math.sqrt(circle_r**2-(veh_width/2)**2)

def collision_check(traj1, traj2):
    time_series = list(traj1.keys())
    for time in time_series:
        center_list_1 = get_circle_center_list(traj1[time])
        center_list_2 = get_circle_center_list(traj2[time])
        for p1 in center_list_1:
            for p2 in center_list_2:
                dist = cal_dist(p1, p2)
                if dist <= 2*circle_r:
                    return True
    return False

def get_circle_center_list(traj_point):
    center1 = (traj_point["x_lon"], traj_point["x_lat"])
    if traj_point["v_lon"] == 0:
        heading = 0
    else:
        heading = math.atan(traj_point["v_lat"]/traj_point["v_lon"])
    center0 = (
        center1[0]+(veh_length/2-tem_len)*math.cos(heading),
        center1[1]+(veh_length/2-tem_len)*math.sin(heading)
    )
    center2 = (
        center1[0]-(veh_length/2-tem_len)*math.cos(heading),
        center1[1]-(veh_length/2-tem_len)*math.sin(heading)
    )
    center_list = [center0, center1, center2]
    return center_list


def cal_dist(p1, p2):
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    return math.sqrt((x1-x2)**2+(y1-y2)**2)


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

class TreeSearchNADEBackgroundController(NADEBackgroundController):
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
        SURROGATE_MODEL_FUNCTION = IDMController.decision_pdf
    PREDICT_MODEL_FUNCTION = IDMController.decision
    # lane_list = np.array([42.0, 46.0, 50.0])
    lane_list = conf.lane_list
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
        new_veh = dict(veh)  # TODO

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
            new_veh, traj = TreeSearchNADEBackgroundController.update_single_vehicle_obs_no_action(veh, duration=duration,
                                                                                     is_lane_change=None, action=action)

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
                        veh_id] = TreeSearchNADEBackgroundController.update_single_vehicle_obs(
                        vehicle, action)
                else:
                    new_full_obs[veh_id] = predicted_full_obs[veh_id][action]
                    trajectory_obs[veh_id] = predicted_full_traj[veh_id][action]
        new_full_obs[cav_id], trajectory_obs[cav_id] = TreeSearchNADEBackgroundController.update_single_vehicle_obs(
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
        cav_obs = TreeSearchNADEBackgroundController.full_obs_to_single_obs(
            new_full_obs, cav_id)
        bv_obs = TreeSearchNADEBackgroundController.full_obs_to_single_obs(
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
        obs["Lead"] = TreeSearchNADEBackgroundController._process_info(
            full_obs, veh_id, longi=1, lateral=0)
        obs["LeftLead"] = TreeSearchNADEBackgroundController._process_info(
            full_obs, veh_id, longi=1, lateral=1)
        obs["RightLead"] = TreeSearchNADEBackgroundController._process_info(
            full_obs, veh_id, longi=1, lateral=-1)
        obs["Foll"] = TreeSearchNADEBackgroundController._process_info(
            full_obs, veh_id, longi=-1, lateral=0)
        obs["LeftFoll"] = TreeSearchNADEBackgroundController._process_info(
            full_obs, veh_id, longi=-1, lateral=1)
        obs["RightFoll"] = TreeSearchNADEBackgroundController._process_info(
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
            TreeSearchNADEBackgroundController.ACTION_NUM)
        depth_flag = (search_depth ==
                      TreeSearchNADEBackgroundController.MAX_TREE_SEARCH_DEPTH)
        CF_flag = TreeSearchNADEBackgroundController.is_CF(cav_obs, bv_obs)[0]
        bv_id = bv_obs["Ego"]["veh_id"]
        if CF_flag:
            challenge_array[2:] = TreeSearchNADEBackgroundController.get_CF_challenge_array(
                cav_obs, bv_obs)
        crash_flag = TreeSearchNADEBackgroundController.crash_check(
            cav_obs, bv_obs, cav_id, bv_id, previous_obs, traj)
        if crash_flag:
            challenge_array = np.ones(
                TreeSearchNADEBackgroundController.ACTION_NUM)
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
            TreeSearchNADEBackgroundController.ACTION_NUM)
        if (not cav_obs) or (not bv_obs):
            cav_obs, bv_obs = TreeSearchNADEBackgroundController.full_obs_to_cav_bv_obs(
                full_obs, cav_id, bv_id)
        leaf_flag, leaf_challenge_array = TreeSearchNADEBackgroundController.leaf_node_check(
            full_obs, previous_obs, traj, cav_obs, bv_obs, cav_id, bv_id, search_depth)
        cav_action_dict, bv_action_dict, cav_pdf, bv_pdf = TreeSearchNADEBackgroundController.get_cav_bv_pdf(
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
                        updated_full_obs, trajectory_obs = TreeSearchNADEBackgroundController.update_obs(
                            full_obs, cav_id, bv_id, bv_action, cav_action, predicted_full_obs, predicted_full_traj)
                        new_challenge_array, updated_bv_pdf = TreeSearchNADEBackgroundController.tree_search_maneuver_challenge(
                            updated_full_obs, full_obs, trajectory_obs, cav_id, bv_id, search_depth + 1)
                        discount_factor = 1
                        if search_depth != 0:
                            discount_factor = conf.treesearch_config["treesearch_discount_factor"]
                        challenge_tmp = discount_factor * np.sum(new_challenge_array * updated_bv_pdf)
                        challenge_tmp_list.append(challenge_tmp)
                    challenge_array[TreeSearchNADEBackgroundController.ACTION_TYPE[bv_action]] += max(
                        challenge_tmp_list)
        return challenge_array, bv_pdf

    @staticmethod
    # @profile
    def get_CF_challenge_array(cav_obs, bv_obs):
        CF_info, bv_v, bv_range_CAV, bv_rangerate_CAV = TreeSearchNADEBackgroundController.is_CF(
            cav_obs, bv_obs)
        if not CF_info:
            raise ValueError("get CF challenge in non-CF mode")
        if CF_info == "CAV_BV":
            return TreeSearchNADEBackgroundController._hard_brake_challenge(bv_v, bv_range_CAV, bv_rangerate_CAV)
        elif CF_info == "BV_CAV":
            return TreeSearchNADEBackgroundController._BV_accelerate_challenge(bv_v, bv_range_CAV, bv_rangerate_CAV)

    @staticmethod
    # @profile
    def get_cav_bv_pdf(cav_obs, bv_obs):
        cav_pdf = TreeSearchNADEBackgroundController.SURROGATE_MODEL_FUNCTION(
            cav_obs)
        _, _, bv_pdf = NDDController.static_get_ndd_pdf(bv_obs)
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
        _, _, bv_pdf = NDDController.static_get_ndd_pdf(bv_obs)
        if full_obs is None:
            full_obs = TreeSearchNADEBackgroundController.cav_bv_obs_to_full_obs(
                bv_obs, CAV)

        # Maneuver challenge calculation
        if utils.is_lane_change(bv_obs["Ego"]) and conf.experiment_config["mode"] != "risk_NDE":
            # if the vehicle is doing lane change, then do not need to calculate the maneuver challenge
            bv_challenge_array, updated_bv_pdf = np.zeros(TreeSearchNADEBackgroundController.ACTION_NUM), np.zeros(
                TreeSearchNADEBackgroundController.ACTION_NUM)
        else:
            bv_challenge_array, updated_bv_pdf = TreeSearchNADEBackgroundController.tree_search_maneuver_challenge(
                full_obs, None, None, CAV["Ego"]["veh_id"], bv_obs["Ego"]["veh_id"], 0, CAV, bv_obs, predicted_full_obs,
                predicted_full_traj)

        # Criticality calculation: maneuver challenge * exposure frequency
        bv_criticality_array = bv_pdf * bv_challenge_array
        risk = np.sum(updated_bv_pdf * bv_challenge_array)
        bv_criticality = np.sum(bv_criticality_array)
        return bv_criticality, bv_criticality_array, bv_challenge_array, risk
