import numpy as np
from .nddcontrollercarla import NDDController
import utils
from conf import conf

class NADEBackgroundControllerCarla(NDDController):
    def __init__(self):
        super().__init__(controllertype="NADEBackgroundController")
        self.weight = None
        self.ndd_possi = None
        self.critical_possi = None
        self.epsilon_pdf_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.normalized_critical_pdf_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.ndd_possi_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.bv_criticality_array = np.zeros(len(conf.ACTIONS), dtype=float)
        self.bv_challenge_array = np.zeros(len(conf.ACTIONS), dtype=float)

    def get_NDD_possi(self):
        if hasattr(self, 'ndd_pdf'):
            print(f"[Debug] get_NDD_possi() returning self.ndd_pdf = {self.ndd_pdf}")
            return self.ndd_pdf
        else:
            print(f"[Debug] get_NDD_possi() called but ndd_pdf not set")
            return [0.0, 1.0, 0.0]

    def _sample_critical_action(self, bv_criticality, criticality_array, ndd_possi_array, epsilon=conf.epsilon_value):
        normalized_critical_pdf_array = criticality_array / bv_criticality
        epsilon_pdf_array = utils.epsilon_greedy(normalized_critical_pdf_array, ndd_possi_array, epsilon=epsilon)
        bv_action_idx = np.random.choice(len(conf.BV_ACTIONS), 1, replace=False, p=epsilon_pdf_array)
        critical_possi, ndd_possi = epsilon_pdf_array[bv_action_idx], ndd_possi_array[bv_action_idx]
        weight_list = (ndd_possi_array + 1e-30) / (epsilon_pdf_array + 1e-30)
        weight = ndd_possi / critical_possi
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
        bv_id = self.vehicle_wrapper.id
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