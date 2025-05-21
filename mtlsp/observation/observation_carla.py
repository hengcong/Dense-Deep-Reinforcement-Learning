from Cython.Shadow import returns
import math
import carla

# TODO： could_drive_adjacent_lane_left / could_drive_adjacent_lane_right, lateral_speed, lateral_offset, prev_action

class ObservationCarla():
    """Observation class store the vehicle observations, the time_stamp object is essential to allow observation to only update once.
    It is composed of the local information, context information, processed information and time stamp.
    local: a dictionary{ vehicle ID: subsribed results (dictionary)
    }
    context: a dictionary{ vehicle ID: subsribed results (dictionary)
    }
    information: a dictionary{
        'Ego': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': 0 [m]},
        'Lead': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'Foll': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'LeftLead': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'RightLead': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'LeftFoll': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
        'RightFoll': {'veh_id': vehicle ID, 'speed': vehicle velocity [m/s], 'position': tuple of X,Y coordinates [m], 'heading': vehicle angle [degree], 'lane_index': lane index of vehicle, 'distance': range between vehicles [m]} or None
    }
    time_stamp is used to record simulation time and for lazy use.
    """
    def __init__(self, veh_id=None, time_stamp=None):
        if not veh_id:
            raise ValueError("No ego vehicle ID is provided!")
        self.veh_id = veh_id
        self.local = None
        self.context = None
        self.information = None
        if time_stamp ==-1:
            raise ValueError("No ego vehicle ID is provided!")
        self.time_stamp = time_stamp

    def update(self, env=None):
        if not env:
            raise ValueError("No environment is provided!")
        elif not env.world:
            raise ValueError("No world is provided!")

        target_vehicle = None
        for veh in [env.ego_vehicle] + env.vehicles:
            if veh.id == self.veh_id:
                target_vehicle = veh
                break

        if target_vehicle is None:
            raise ValueError(f"Vehicle with ID {self.veh_id} not found!")

        self.time_stamp = env.world.get_snapshot().timestamp.elapsed_seconds
        self.local = {self.veh_id: self._get_vehicle_observation(target_vehicle, env.world)}
        self.context = {}

        self.information = self._process_observation(env, target_vehicle)

    def _get_vehicle_observation(self, vehicle, world, ego_pos3d=None):
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        waypoint = world.get_map().get_waypoint(transform.location, project_to_road=True,
                                                lane_type=carla.LaneType.Driving)

        if ego_pos3d is None:
            distance = 0.0
        else:
            dx = transform.location.x - ego_pos3d[0]
            dy = transform.location.y - ego_pos3d[1]
            dz = transform.location.z - ego_pos3d[2]
            distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        heading = transform.rotation.yaw
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        acc = math.sqrt(acceleration.x ** 2 + acceleration.y ** 2 + acceleration.z ** 2)

        forward_vector = transform.get_forward_vector()
        right_vector = carla.Vector3D(-forward_vector.y, forward_vector.x, 0.0)
        lateral_speed = velocity.x * right_vector.x + velocity.y * right_vector.y

        can_left, can_right = False, False
        if waypoint:
            left_wp = waypoint.get_left_lane()
            right_wp = waypoint.get_right_lane()
            if left_wp and left_wp.lane_type == carla.LaneType.Driving and left_wp.lane_id * waypoint.lane_id > 0:
                can_left = True
            if right_wp and right_wp.lane_type == carla.LaneType.Driving and right_wp.lane_id * waypoint.lane_id > 0:
                can_right = True

        return {
            'veh_id': vehicle.id,
            'position': (transform.location.x, transform.location.y),
            'position3d': (transform.location.x, transform.location.y, transform.location.z),
            'speed': speed,
            'acceleration': acc,
            'heading': heading,
            'lane_index': waypoint.lane_id if waypoint else None,
            'road_id': waypoint.road_id if waypoint else None,
            'distance': distance,
            'lateral_speed': lateral_speed,
            'lateral_offset': 0.0,
            'could_drive_adjacent_lane_left': can_left,
            'could_drive_adjacent_lane_right': can_right,
            'prev_action': getattr(vehicle, "last_action", None),
        }

    def _process_observation(self, env=None, target_vehicle=None):
        # get surrounding vehicle information
        surrounding = env.get_surrounding_vehicles(target_vehicle)
        world = env.world
        ego_pos3d = self.local[self.veh_id]['position3d']

        processed_info = {'Ego': self.local[self.veh_id],}
        for key in ['Lead', 'LeftLead', 'RightLead', 'Foll', 'LeftFoll', 'RightFoll']:
            vehicle = surrounding.get(key)
            if vehicle is not None:
                processed_info[key] = self._get_vehicle_observation(vehicle, world, ego_pos3d)
            else:
                processed_info[key] = None
        # functions finding the surrounding vehicles
        return processed_info


