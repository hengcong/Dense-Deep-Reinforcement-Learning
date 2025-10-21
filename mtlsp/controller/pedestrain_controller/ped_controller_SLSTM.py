from abc import ABC, abstractmethod
import carla
import math
from mtlsp.observation.pedestrian_observation_carla_v1 import PedestrianObservationCarla
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
    obs_len = 20
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
        self.action = None
        self.max_skip = 2 #NOTE
        self.gap_reset = 8 #NOTE

    def step(self):

        # 1) get frame index robustly
        world = getattr(self.pedestrian_wrapper, "world", None)
        if world is  None:
            control = self._stop_control()
            self.action = control
            return control
        snapshot = world.get_snapshot()
        if snapshot is None:
            control = self._stop_control()
            self.action = control
            return control
        frame_idx = int(snapshot.frame)

        # 2) Load data from observation (new signature + ready flag)
        traj_pack = self._read_and_process_trajectory(frame_idx)
        # traj_pack -> (traj_data, ego_id, ego_curr_pos, last_frame, ready)
        self.traj_data, ego_id, ego_curr_pos, last_frame, ready = traj_pack

        if not ready or ego_id is None or ego_curr_pos is None:
            control = self._stop_control()
            self.action = control
            return control

        curr_x, curr_y = ego_curr_pos

        # 3) Get prediction trajectory (use .get to avoid KeyError)
        predicted_traj = predict_trajectory(self.traj_data)
        if isinstance(predicted_traj, list):
            # assume only ego traj predicted, wrap it into dict
            predicted_traj = {ego_id: predicted_traj}
        # {pid: [(frame,x,y), ...], ...}
        pred_traj_list = predicted_traj.get(ego_id, [])
        if pred_traj_list and len(pred_traj_list[0]) == 4:
            pred_traj_list = [t for t in pred_traj_list if t[1] == ego_id]
        if not pred_traj_list:
            control = self._stop_control()
            self.action = control
            return control

        #ego_pred_traj = [t for t in pred_traj_list if t[0] > last_frame]
        SAMPLE_STEP = 10
        ego_pred_traj = [t for i, t in enumerate(pred_traj_list) if i % SAMPLE_STEP == 0 and t[0] > last_frame]

        ego_pred_traj.sort(key=lambda a: a[0])
        if not ego_pred_traj:
            control = self._stop_control()
            self.action = control
            return control

        control = self._stop_control()
        self.action = control

        LOOKAHEAD_FRAMES = 5
        far_pred = [t for t in ego_pred_traj if (t[0] - last_frame) >= LOOKAHEAD_FRAMES]
        if far_pred:
            ego_pred_traj = far_pred
        else:
            ego_pred_traj = ego_pred_traj[-1:]

        # 4) Calculate speed & direction (guard dt)
        frame,_, pred_x, pred_y = ego_pred_traj[0]
        raw_steps = max(1, int(frame - last_frame))
        dt = raw_steps * self.frame_dt

        dx, dy = pred_x - curr_x, pred_y - curr_y
        distance = math.hypot(dx, dy)
        if distance < 1e-4:
            control = self._stop_control()
            self.action = control
            return control

        direction = carla.Vector3D(dx / distance, dy / distance, 0.0)
        speed = min(self.max_speed, distance / dt)
        # 5) Send command signal to CARLA and RETURN it
        control = carla.WalkerControl()
        control.direction = direction
        control.speed = speed
        control.jump = False
        self.action = control
        return control
    # def step(self):
    #     # Load data from observation
    #
    #     self.traj_data, ego_id, ego_curr_pos, last_frame = self._read_and_process_trajectory()
    #     if ego_id is None:
    #         control = self._stop_control()
    #         return control
    #     curr_x, curr_y = ego_curr_pos
    #
    #     # Get prediction trajectory
    #     predicted_traj = predict_trajectory(self.traj_data)     # a dictionary of prediction list({pid:[frame, x, y], ...})
    #     pred_traj_list = predicted_traj[ego_id]
    #     if not pred_traj_list:
    #         control = self._stop_control()
    #         return control
    #     ego_pred_traj = list(filter(lambda x: x[0] > last_frame, pred_traj_list))
    #     ego_pred_traj = sorted(ego_pred_traj, key=lambda a: a[0])       # sort the prediction traj using frame
    #     if not ego_pred_traj:
    #         control = self._stop_control()
    #         return control
    #
    #     # Calcuate speed, direction for control signal
    #     frame, pred_x, pred_y = ego_pred_traj[0]
    #     dt = min(1, (frame-last_frame)) * self.frame_dt
    #     dx, dy = pred_x-curr_x, pred_y-curr_y
    #     distance = math.sqrt(dx**2 + dy**2)
    #     if distance < 0.0001:
    #         control = self._stop_control()
    #         return control
    #     normalized_dir = [float(dx/distance), float(dy/distance), 0.0]      # get normalized direction
    #     direction = carla.Vector3D(*normalized_dir)
    #     speed = math.sqrt((pred_x-curr_x)**2 + (pred_y-curr_y)**2)/dt
    #     speed = min(self.max_speed, speed)
    #
    #     # Send command signal to CARLA
    #     control = carla.WalkerControl()
    #     control.direction = direction
    #     control.speed = speed
    #     control.jump = False        # change if needed
    #     self.action = control
    #     return

    def _read_and_process_trajectory(self, frame_idx: int):
        """
        Returns:
            traj_data: List[Tuple[frame, ped_id, x, y]]
            ego_id: int | None
            ego_curr_pos: Tuple[float, float] | None
            last_frame: int
            ready: bool
        """
        info = getattr(self.pedestrian_wrapper.observation, "information", None)
        if not info or not isinstance(info, dict):
            # observation not ready on this frame
            return [], None, None, -1, False

        ego_id = info.get("ped_id")
        ego_traj = info.get("trajectory") or []
        ego_curr_pos = info.get("position")
        neighbors_id = info.get("neighbors_ids") or []
        neighbors_traj = info.get("neighbors_trajectory") or {}

        # Bootstrap first frame: if no ego trajectory yet but we have current position
        if not ego_traj and ego_curr_pos is not None and ego_id is not None:
            try:
                ex, ey = float(ego_curr_pos[0]), float(ego_curr_pos[1])
            except Exception:
                ex, ey = None, None
            if ex is not None and ey is not None:
                ego_traj = [(frame_idx, ex, ey)]

        traj_data = []
        last_frame = -1
        ego_frames = []

        # Ego
        for item in ego_traj:
            # robust unpack (item may be tuple/list of length 3)
            try:
                frame, x, y = item
            except Exception:
                continue
            traj_data.append((int(frame), ego_id, float(x), float(y)))
            last_frame = max(last_frame, int(frame))
            ego_frames.append(int(frame))

        # Neighbors
        for nid in neighbors_id:
            n_traj = neighbors_traj.get(nid) or []
            for nitem in n_traj:
                try:
                    nframe, nx, ny = nitem
                except Exception:
                    continue
                traj_data.append((int(nframe), nid, float(nx), float(ny)))
                last_frame = max(last_frame, int(nframe))

        traj_data.sort(key=lambda r: (r[0], r[1]))
        ready = False
        if len(ego_frames) >= self.obs_len:
            ego_frames_sorted = sorted(set(ego_frames)) # NOTE
            coverage_ok = (ego_frames_sorted[-1] - ego_frames_sorted[0]) > (self.obs_len -1)
            gaps = [ego_frames_sorted[i+1] - ego_frames_sorted[i] for i in range(len(ego_frames_sorted) - 1)]
            max_gap = max(gaps) if gaps else 0
            continuity_ok = max_gap <=self.gap_reset
            reset_needed = max_gap >= self.gap_reset
            if reset_needed:
                ready = False
            else:
                 ready = coverage_ok and continuity_ok

        return traj_data, ego_id, ego_curr_pos, last_frame, ready

    # # Process the file (sort the data, make it more readable)
    # def _read_and_process_trajectory(self):
    #     observation = self.pedestrian_wrapper.observation.information
    #     ego_id = observation['ped_id']
    #     ego_traj = observation['trajectory']
    #     ego_curr_pos = observation['position']
    #     neighbors_id = observation['neighbors_ids']
    #     neighbors_traj = observation['neighbors_trajectory']        # a dictionary
    #
    #     traj_data = []
    #     last_frame = -1
    #     # put ego traj into traj_data
    #     for frame, x, y in ego_traj:
    #         traj_data.append((frame, ego_id, x, y))
    #         last_frame = max(last_frame, frame)
    #
    #     # put neighbors traj into traj_data
    #     for nid in neighbors_id:
    #         if nid not in neighbors_traj.keys():
    #             continue
    #         n_traj = neighbors_traj[nid]
    #         for nframe, nx, ny in n_traj:
    #             traj_data.append((nframe, nid, nx, ny))
    #     traj_data.sort(key=lambda r: (r[0], r[1]))      # sort according to frame, ped_id
    #
    #     return traj_data, ego_id, ego_curr_pos, last_frame

    def _stop_control(self):
        control = carla.WalkerControl()
        control.direction = carla.Vector3D(0.0, 0.0, 0.0)
        control.speed = 0.0
        control.jump = False
        return control
