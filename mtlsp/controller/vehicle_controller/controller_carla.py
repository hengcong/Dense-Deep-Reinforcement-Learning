from abc import ABC, abstractmethod
import numpy as np
from bidict import bidict
from mtlsp.observation.observation_carla import ObservationCarla

class Controller_Carla(ABC):
    def __init__(self, observation_method=None, controllertype="DummyController"):
        self.ego_info = None
        self.action = None
        self._type = controllertype
        self.observation_method = observation_method
        self.control_log = {}
        self.vehicle_wrapper = None

    def attach_to_vehicle(self, vehicle_wrapper):
        self.vehicle_wrapper = vehicle_wrapper

    def reset(self):
        pass

    def step(self):
        pass

    @property
    def type(self):
        """Return controller type.

        Returns:
            str: Controller type.
        """
        return self._type

class DiscreetController(Controller_Carla):
    longi_safety_buffer, lateral_safety_buffer = 2, 2
    v_low, v_high, r_low, r_high, rr_low, rr_high, acc_low, acc_high =20, 40, 0, 115, -10, 8, -4, 2
    acc_resolution = 0.2
    LENGTH = 5
    ACTION_STEP = 1.0
    num_acc = int(1+((acc_high-acc_low)/acc_resolution))
    CAV_acc_low, CAV_acc_high, CAV_acc_step = -4, 2, 0.2
    num_CAV_acc = int((CAV_acc_high - CAV_acc_low)/CAV_acc_step + 1)
    CAV_acc_to_idx_dic = bidict()
    for i in range(num_CAV_acc): CAV_acc_to_idx_dic[list(np.arange(CAV_acc_low, CAV_acc_high + CAV_acc_step, CAV_acc_step))[i]] = i
    acc_to_idx_dic = bidict()
    for m in range(num_acc): acc_to_idx_dic[list(np.linspace(acc_low, acc_high, num=num_acc))[m]] = m

    def __init__(self, observation_method = ObservationCarla, controllertype="DiscreetController"):
        super().__init__(controllertype=controllertype,observation_method=observation_method)

    def step(self):
        """ store ego vehicle information."""
        self.ego_info = self.vehicle_wrapper.observation.information["Ego"]

    @staticmethod
    def _check_longitudinal_safety(obs, pdf_array, lateral_result=None, CAV_flag=False): # NOTE:Although CAV_flag is set to False by default, no actual value is passed in during subsequent calls, which means all vehicles are treated as background vehicles (BV), even when they are actually CAVs.
        """Check longitudinal safety feasibility based on relative speed and distance."""
        ego_info = obs["Ego"]
        f_veh_info = obs["Lead"]
        safety_buffer = DiscreetController.longi_safety_buffer

        for i in range(len(pdf_array) - 1, -1, -1):
            if CAV_flag:
                acc = DiscreetController.CAV_acc_to_idx_dic.inverse[i]
            else:
                acc = DiscreetController.acc_to_idx_dic.inverse[i]

            if f_veh_info is not None:
                rr = f_veh_info["speed"] - ego_info["speed"]
                r = f_veh_info["distance"]
                criterion_1 = rr + r + 0.5 * (DiscreetController.acc_low - acc)

                self_v_2 = max(ego_info["speed"] + acc, DiscreetController.v_low)
                f_v_2 = max(f_veh_info["speed"] + DiscreetController.acc_low, DiscreetController.v_low)

                dist_r = (self_v_2 ** 2 - DiscreetController.v_low ** 2) / (2 * abs(DiscreetController.acc_low))
                dist_f = (
                        (f_v_2 ** 2 - DiscreetController.v_low ** 2) / (2 * abs(DiscreetController.acc_low))
                        + DiscreetController.v_low * (f_v_2 - self_v_2) / DiscreetController.acc_low
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
        safety_buffer = DiscreetController.lateral_safety_buffer

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
                    dis_change = rr * DiscreetController.ACTION_STEP + 0.5 * DiscreetController.acc_low * (DiscreetController.ACTION_STEP ** 2)
                    r_1 = r + dis_change
                    rr_1 = rr + DiscreetController.acc_low * DiscreetController.ACTION_STEP

                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0
                    elif rr_1 < 0:
                        self_v_2 = max(ego_info["speed"], DiscreetController.v_low)
                        f_v_2 = max(f_veh["speed"] + DiscreetController.acc_low, DiscreetController.v_low)
                        dist_r = (self_v_2 ** 2 - DiscreetController.v_low ** 2) / (2 * abs(DiscreetController.acc_low))
                        dist_f = (f_v_2 ** 2 - DiscreetController.v_low ** 2) / (2 * abs(DiscreetController.acc_low)) + DiscreetController.v_low * (f_v_2 - self_v_2) / DiscreetController.acc_low
                        r_2 = r_1 - dist_r + dist_f
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0

                if r_veh is not None:
                    rr = ego_info["speed"] - r_veh["speed"]
                    r = r_veh["distance"]
                    dis_change = rr * DiscreetController.ACTION_STEP - 0.5 * DiscreetController.acc_high * (DiscreetController.ACTION_STEP ** 2)
                    r_1 = r + dis_change
                    rr_1 = rr - DiscreetController.acc_high * DiscreetController.ACTION_STEP

                    if r_1 <= safety_buffer or r <= safety_buffer:
                        pdf_array[lane_index] = 0
                    elif rr_1 < 0:
                        self_v_2 = min(ego_info["speed"], DiscreetController.v_high)
                        r_v_2 = min(r_veh["speed"] + DiscreetController.acc_high, DiscreetController.v_high)
                        dist_r = (r_v_2 ** 2 - DiscreetController.v_low ** 2) / (2 * abs(DiscreetController.acc_low))
                        dist_f = (self_v_2 ** 2 - DiscreetController.v_low ** 2) / (2 * abs(DiscreetController.acc_low)) + DiscreetController.v_low * (-r_v_2 + self_v_2) / DiscreetController.acc_low
                        r_2 = r_1 - dist_r + dist_f
                        if r_2 <= safety_buffer:
                            pdf_array[lane_index] = 0

        if np.sum(pdf_array) == 0:
            return np.array([0, 1, 0])

        if self.vehicle_wrapper.id == ego_info.get("veh_id"):
            return pdf_array
        else:
            return pdf_array / np.sum(pdf_array)

class ContinuousController(Controller_Carla):
    def __init__(self):
        super().__init__(controllertype="ContinuousController")