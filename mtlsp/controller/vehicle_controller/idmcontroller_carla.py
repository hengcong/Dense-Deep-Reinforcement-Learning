import numpy as np
import carla

CAV_POLITENESS = 0.0
CAV_LANE_CHANGE_MIN_GAIN = 0.1
CAV_LANE_CHANGE_MAX_BRAKING = 4.0
ACC_LO = -4.0
ACC_HI = 2.0
VEHICLE_LENGTH = 5.0

CAV_PARAMS = {
    "v_desired": 33.33,
    "a_max": 1.5,
    "acc_min": -2.0,
    "d0": 2.0,
    "tau": 1.2,
    "delta": 4.0
}

BV_PARAMS = {
    "v_desired": 35.0,
    "a_max": 2.0,
    "acc_min": -4.0,
    "d0": 5.0,
    "tau": 1.5,
    "delta": 4.0
}

class IDMController:
    def __init__(self):
        self.control_log = {}
        self.vehicle_wrapper = None
        self.last_action = None

    def attach_to_vehicle(self, wrapper):
        self.vehicle_wrapper = wrapper

    def reset(self):
        self.last_action = None

    def step(self):
        observation = self.vehicle_wrapper.observation.information
        ego = observation["Ego"]
        front = observation.get("Lead")
        left_lead = observation.get("LeftLead")
        left_foll = observation.get("LeftFoll")
        right_lead = observation.get("RightLead")
        right_foll = observation.get("RightFoll")

        veh_type = "CAV" if ego.get("veh_id") == self.vehicle_wrapper.vehicle.id else "BV"
        params = CAV_PARAMS if veh_type == "CAV" else BV_PARAMS

        lc_decision = self.mobil_decision(ego, left_lead, left_foll, right_lead, right_foll, front, params)
        if lc_decision:
            action = lc_decision
        else:
            acc = self.idm_acc(ego, front, params)
            acc = np.clip(acc, ACC_LO, ACC_HI)
            action = {"lateral": "central", "longitudinal": acc}

        self.last_action = action

        control = carla.VehicleControl()
        acc = action["longitudinal"]
        control.throttle = np.clip(acc / 3.0, 0.0, 1.0) if acc > 0 else 0.0
        control.brake = np.clip(-acc / 8.0, 0.0, 1.0) if acc < 0 else 0.0
        control.steer = {
            "left": -0.3,
            "right": 0.3
        }.get(action.get("lateral", "central"), 0.0)

        self.vehicle_wrapper.vehicle.apply_control(control)
        return control

    def idm_acc(self, ego, front, params):
        v = ego["speed"]
        v0 = params["v_desired"]
        a_max = params["a_max"]
        delta = params["delta"]

        acc = a_max * (1 - (v / v0) ** delta)
        if front:
            s = front["distance"] - VEHICLE_LENGTH
            s = max(s, 0.1)
            s_star = self.desired_gap(ego, front, params)
            acc -= a_max * (s_star / s) ** 2
        return acc

    def desired_gap(self, ego, front, params):
        d0 = params["d0"]
        tau = params["tau"]
        a_max = params["a_max"]
        b_comf = -params["acc_min"]
        dv = ego["speed"] - front["speed"]
        return d0 + max(0, ego["speed"] * tau + ego["speed"] * dv / (2 * np.sqrt(a_max * b_comf)))

    def mobil_decision(self, ego, ll, lf, rl, rf, front, params):
        options = [("left", ll, lf), ("right", rl, rf)]
        best_gain = -np.inf
        best_side = None

        for side, lead, foll in options:
            if lead and lead["distance"] <= 0:
                continue
            if foll and foll["distance"] <= 0:
                continue

            new_foll_acc = self.idm_acc(foll, ego, params) if foll else 0
            new_foll_orig = self.idm_acc(foll, lead, params) if foll and lead else 0
            ego_orig = self.idm_acc(ego, front, params)
            ego_new = self.idm_acc(ego, lead, params)

            if new_foll_acc < -CAV_LANE_CHANGE_MAX_BRAKING:
                continue

            gain = ego_new - ego_orig + CAV_POLITENESS * (new_foll_acc - new_foll_orig)
            if gain > CAV_LANE_CHANGE_MIN_GAIN and gain > best_gain:
                best_gain = gain
                best_side = side

        if best_side:
            return {"lateral": best_side, "longitudinal": 0.0}
        return None
