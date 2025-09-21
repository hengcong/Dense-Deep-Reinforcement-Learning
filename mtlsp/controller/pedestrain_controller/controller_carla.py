from abc import ABC, abstractmethod
import numpy as np
from bidict import bidict
from mtlsp.observation.pedestrian_observation_carla import PedestrianObservationCarla
import os
import pickle
import torch
import carla
import math
from collections import defaultdict
from mtlsp.pedestrian.SLSTM.SocialLSTM import SocialModel
from mtlsp.pedestrian.SLSTM.traj_prediction import predict_trajectory

class ControllerCarla(ABC):
    def __init__(self, observation_method=None, controllertype="DummyController"):
        self._type = controllertype
        self.observation_method = observation_method
        self.control_log = {}

    def attach_to_pedestrain(self, pedestrian_wrapper):
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
    

'''
PedestrianController class:
1. Gets the input from the observation (PedestrianObservationCarla class) and process the information 
2. Feed the data to S-LSTM to predict the following trajectory of pedestrian
3. Send command signal to pedestrian models in CARLA
'''
class PedestrianController(ControllerCarla):
    def __init__(self, observation_method=PedestrianObservationCarla, controllertype="PedestrianController"):
        super().__init__(observation_method=observation_method, controllertype=controllertype)

        self.traj_data = None
        self.pedestrian_wrapper = None

        # Path to pedestrian trajectory: cwd/mtlsp/pedestrian/pedestrian_trajectory_raw.txt
        cwd = os.getcwd()
        subdir = 'mtlsp/pedestrian'
        raw_traj_name = 'pedestrian_trajectory_raw.txt'
        sorted_traj_name = 'pedestrian_trajectory_clean.txt'
        self.save_dir = os.path.join(cwd, subdir)
        self.raw_traj_path = os.path.join(self.save_dir, raw_traj_name)
        self.sorted_traj_path = os.path.join(self.save_dir, sorted_traj_name)

    
    def step(self):
        # Read and process pedestrian_trajectory_raw.txt
        self.traj_data = self.read_and_process_trajectory()



        # Get prediction
        predicted_traj = predict_trajectory(self.traj_data)

        # Send command signal to CARLA
        control = []
        last_frame = 0
        for frame_id, ped_id, x, y in predicted_traj:
            direction = [x, y, 0]
            speed = math.sqrt(x**2 + y**2)/(frame_id - last_frame)
            control = carla.WalkerControl()
            control.direction = carla.Vector3D(*direction)
            control.speed = speed
            control.jump = False        # change if needed

            # Apply control to the pedestrian model
            if ped_id in self.pedestrian_wrapper:
                self.pedestrian_wrapper[ped_id].apply_control(control)
            else:
                print(f"Warning: pedestrian {ped_id} not found in wrapper.")
        
            last_frame = frame_id    

        # Update pedestrian_trajectory_raw.txt
        self.update_trajectory(predicted_traj)

        # Return the control siganl to individual pedestrian
        return control


    # Process the file (sort the data, make it more readable)
    def read_and_process_trajectory(self):
        processed_info = []
        with open(self.raw_traj_path, 'r') as f:
            for line in f:
                frame_id, ped_id, x, y = map(float, line.strip().split())
                processed_info.append((frame_id, ped_id, x, y))
        processed_info.sort(key=lambda tup: (tup[1], tup[0]))

        # Create a sorted file (pedestrian_trajectory_clean.txt)
        with open(self.sorted_traj_path, 'w') as f:
            for frame_id, ped_id, x, y in processed_info:
                f.write(f"{frame_id} {ped_id} {x:.3f} {y:.3f}\n")

        return processed_info



    def update_trajectory(self, predicted_traj):
        '''
        Add the predicted path to pedestrian_trajectory_raw.txt
        '''
        if predicted_traj is None:
            print("Warning: No predicted trajectory to update.")
            return

        with open(self.raw_traj_path, 'a') as f:
            for frame_id, ped_id, x, y in predicted_traj:
                f.write(f"{int(frame_id)} {int(ped_id)} {x:.3f} {y:.3f}\n")
        
