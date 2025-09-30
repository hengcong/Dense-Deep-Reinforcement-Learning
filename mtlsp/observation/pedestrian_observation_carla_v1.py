from Cython.Shadow import returns
import math
import carla
import os

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

class PedestrianObservationCarla:
    def __init__(self, target_ped_id=None, time_stamp=None):
        self.information = {}
        self.target_ped_id = target_ped_id
        self.trajectory = []

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


        # Get the information for each pedestrian
        ped_list = env.get_actors().filter('walker.pedestrian.*')
        for num_ped in range(len(ped_list)):
            ped = ped_list[num_ped]
            obs = self._get_ped_observation(pedestrian=ped)
            ped_id = ped.id

            # Update trajectory
            frame_id = self.snapshot.frame
            x, y = obs['position']
            self.trajectory.append((frame_id, ped_id, x, y))

            if ped.id == self.target_ped_id:
                self.information["Target"] = obs
            else:
                self.information[f"Ped_{num_ped}"] = obs
        
        # Save the pedestrian information
        self.trajectory_saver()

        # Clear pedestrian information in this frame
        self.trajectory.clear()
        
        
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
    

    def trajectory_saver(self):
        '''
        Save pedestrian trajectory to cwd/mtlsp/pedestrian/pedestrian_trajectory_raw.txt
        '''
        cwd = os.getcwd()
        subdir = 'mtlsp/pedestrian'
        save_dir = os.path.join(cwd, subdir)
        os.makedirs(save_dir, exist_ok=True)
        file_name = 'pedestrian_trajectory_raw.txt'
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, 'a') as f:
            for frame_id, ped_id, x, y in self.trajectory:
                f.write(f"{frame_id} {ped_id} {x:.3f} {y:.3f}\n")
