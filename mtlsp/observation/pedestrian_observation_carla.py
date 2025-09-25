import math
import carla
from mtlsp.pedestrian.ped_obs_utils import TrajStore

'''
The PedestrianObservationCarla class get all information of pedestrian in CARLA.
(Hint: the camera is BEV focus on one of the pedestrians)
information: a dictionary:{
                            'ped_id': pedestrian ID,
                            'position': a tuple of current X, Y coordinates,
                            'position3d': a tuple of current X, Y, Z coordinates,
                            'speed': pedestrian velocity [m/s],
                            'acceleration': pedestrian acceleration,
                            'trajectory': a list of (frame, x, y) tuples of past trajectory,
                            'neighbors_ids': a list of neighbor pedestrian IDs,
                            'neighbors_trajectory': a dictionary of neighbor pedestrian trajectories {ped_id: [(frame, x, y)]}
                        },
'''

class PedestrianObservationCarla():
    def __init__(self, target_ped_id=None, time_stamp=None, traj_store: TrajStore = None):
        self.information = {}
        self.target_ped_id = target_ped_id
        self.traj_store = traj_store

        if time_stamp ==-1:
            raise ValueError("No target pedestrian ID is provided!")
        self.time_stamp = time_stamp
        self.frame = None


    
    def update(self, env=None):
        if not env:
            raise ValueError("No environment is provided!")
        elif not env.world:
            raise ValueError("No world is provided!")
        
        snapshot = env.world.get_snapshot()
        self.frame  = int(snapshot.frame)
        self.time_stamp = float(snapshot.timestamp.elapsed_seconds)
        
        # get information for the target pedestrian
        self.traj_store.scan_if_needed(env)
        self.information = self._get_ped_observation(env, self.target_ped_id)

        
    def _get_ped_observation(self, env, ped_id):
        '''
        Get all past and current information for pedestrian
        '''
        if ped_id is None:
            raise ValueError("No pedestrian ID is provided!")

        pedestrian = env.world.get_actor(ped_id)
        transform = pedestrian.get_transform()
        velocity = pedestrian.get_velocity()
        acceleration = pedestrian.get_acceleration()
    
        heading = transform.rotation.yaw
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        acc = math.sqrt(acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2)

        obs = self.traj_store.build_observation(ped_id, relative=False)     # get the global coordinates for every neighbors (True if relative coordinates are needed)
        ego_traj = obs["Ego Past Trajectory"]
        neighbors_id = obs["Neighbors"]
        neighbors_past_traj = obs["Neighbors' Past Trajectories"]   # a dictionary


        return {
            'ped_id':ped_id,
            'position': (transform.location.x, transform.location.y),
            'position3d': (transform.location.x, transform.location.y, transform.location.z),
            'speed': speed,
            'acceleration': acc,
            'heading': heading,
            'trajectory': ego_traj,
            'neighbors_ids': neighbors_id,
            'neighbors_trajectory': neighbors_past_traj
        }
    

