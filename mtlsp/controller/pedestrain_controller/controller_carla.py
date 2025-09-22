from abc import ABC, abstractmethod
import carla
import math
from mtlsp.observation.pedestrian_observation_carla import PedestrianObservationCarla
from mtlsp.pedestrian.SLSTM.traj_prediction import predict_trajectory

class ControllerCarla(ABC):
    def __init__(self, observation_method=None, controllertype="DummyController"):
        self._type = controllertype
        self.observation_method = observation_method
        self.control_log = {}

    def attach_to_pedestrian(self, pedestrian_wrapper):
        self.pedestrian_wrapper = pedestrian_wrapper

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
    


class PedestrianController(ControllerCarla):
    '''
    PedestrianController class:
    1. Gets the input from the observation (PedestrianObservationCarla class) and process the information 
    2. Feed the data to S-LSTM to predict the following trajectory of pedestrian
    3. Send command signal to pedestrian models in CARLA
    '''
    def __init__(self, observation_method=PedestrianObservationCarla, controllertype="PedestrianController"):
        super().__init__(observation_method=observation_method, controllertype=controllertype)

        self.traj_data = None
        self.pedestrian_wrapper = None
        self.max_speed = 1.4    # limit for carla (m/s)
        self.frame_dt = 0.05    # time interval for every frames

    def step(self):
        # Load data from observation
        self.traj_data, ego_id, ego_curr_pos, last_frame = self._read_and_process_trajectory()
        if ego_id is None:
            control = self._stop_control()
            return control
        curr_x, curr_y = ego_curr_pos

        # Get prediction trajectory
        predicted_traj = predict_trajectory(self.traj_data)     # a dictionary of prediction list({pid:[frame, x, y], ...})
        pred_traj_list = predicted_traj[ego_id]
        if not pred_traj_list:
            control = self._stop_control()
            return control
        ego_pred_traj = list(filter(lambda x: x[0] > last_frame, pred_traj_list))
        ego_pred_traj = sorted(ego_pred_traj, key=lambda a: a[0])       # sort the prediction traj using frame
        if not ego_pred_traj:
            control = self._stop_control()
            return control

        # Calcuate speed, direction for control signal
        frame, pred_x, pred_y = ego_pred_traj[0]
        dt = min(1, (frame-last_frame)) * self.frame_dt
        dx, dy = pred_x-curr_x, pred_y-curr_y
        distance = math.sqrt(dx**2 + dy**2)
        if distance < 0.0001:
            control = self._stop_control()
            return control
        normalized_dir = [float(dx/distance), float(dy/distance), 0.0]      # get normalized direction
        direction = carla.Vector3D(*normalized_dir)
        speed = math.sqrt((pred_x-curr_x)**2 + (pred_y-curr_y)**2)/dt
        speed = min(self.max_speed, speed)

        # Send command signal to CARLA
        control = carla.WalkerControl()
        control.direction = direction
        control.speed = speed
        control.jump = False        # change if needed
  
        return control


    # Process the file (sort the data, make it more readable)
    def _read_and_process_trajectory(self):
        observation = self.pedestrian_wrapper.observation.information
        ego_id = observation['ped_id']
        ego_traj = observation['trajectory']
        ego_curr_pos = observation['position']
        neighbors_id = observation['neighbors_ids']
        neighbors_traj = observation['neighbors_trajectory']        # a dictionary

        traj_data = []
        last_frame = -1
        # put ego traj into traj_data
        for frame, x, y in ego_traj:
            traj_data.append((frame, ego_id, x, y))
            last_frame = max(last_frame, frame)

        # put neighbors traj into traj_data
        for nid in neighbors_id:
            if nid not in neighbors_traj.keys():
                continue
            n_traj = neighbors_traj[nid]
            for nframe, nx, ny in n_traj:
                traj_data.append((nframe, nid, nx, ny))
        traj_data.sort(key=lambda r: (r[0], r[1]))      # sort according to frame, ped_id

        return traj_data, ego_id, ego_curr_pos, last_frame

    def _stop_control(self):
        control = carla.WalkerControl()
        control.direction = carla.Vector3D(0.0, 0.0, 0.0)
        control.speed = 0.0
        control.jump = False
        return control
