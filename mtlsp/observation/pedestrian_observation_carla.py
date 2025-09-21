from Cython.Shadow import returns
import math
import carla
import os
from mtlsp.pedestrian.ped_obs_service import Ped_Obs_Service

'''
The PedestrianObservationCarla class get all information of pedestrian in CARLA.
(Hint: the camera is BEV focus on one of the pedestrians)
information: a dictionary:{
                            'ped_id': pedestrian ID,
                            'position': a tuple of current X, Y coordinates,
                            'position3d': a tuple of current X, Y, Z coordinates,
                            'speed': pedestrian velocity [m/s],
                            'acceleration': pedestrian acceleration,
                        }, ...
'''

class PedestrianObservationCarla():
    def __init__(self, target_ped_id=None, time_stamp=None, traj_store: Ped_Obs_Service = None):
        self.information = {}
        self.target_ped_id = target_ped_id
        self.trajectory_storage = traj_store

        if time_stamp ==-1:
            raise ValueError("No target pedestrian ID is provided!")
        self.time_stamp = time_stamp
        self.snapshot = None


    
    def update(self, env=None):
        if not env:
            raise ValueError("No environment is provided!")
        elif not env.world:
            raise ValueError("No world is provided!")
        
        # Get information from current frame
        self.time_stamp = env.world.get_snapshot().timestamp.elapsed_seconds
        self.snapshot = env.world.get_snapshot()


        # Implement build_observtion here
        
        
    def _get_ped_observation(self, pedestrian):
        '''
        Get all past and current information for pedestrian
        '''
        transform = pedestrian.get_transform()
        velocity = pedestrian.get_velocity()
        acceleration = pedestrian._get_acceleration()
    
        heading = transform.rotation.yaw
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        acc = math.sqrt(acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2)


        return {
            'ped_id':pedestrian.id,
            'position': (transform.location.x, transform.location.y),
            'position3d': (transform.location.x, transform.location.y, transform.location.z),
            'speed': speed,
            'acceleration': acc,
            'heading': heading,
        }
    

