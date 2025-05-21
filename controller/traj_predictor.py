from cmath import isclose
import carla
from controller.traj_utils import TrajUtils
import math
import conf.conf as conf

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

class Traj:
    def __init__(self,x_lon, x_lat, v_lon, v_lat, lane_index, lane_list_info = conf.lane_list, carla_map = None): # TODO: lane_list_info
        self.traj_info = {}
        self.lane_list_info = lane_list_info if lane_list_info is not None else conf.lane_list
        self.traj_info["0.0"] = {"x_lon": x_lon, "x_lat": x_lat, "v_lon": v_lon, "v_lat": v_lat, "lane_index": lane_index}
        self.carla_map = carla_map


    def crop(self, start_time, end_time):
        pop_key_list = []
        for time in self.traj_info:
            if (float(time) < start_time and not isclose(start_time, float(time))) or (float(time) > end_time and not isclose(end_time, float(time))):
                pop_key_list.append(time)
        for key in pop_key_list:
            self.traj_info.pop(key)

    def predict_with_action(self, action, start_time=0.0, time_duration=1.0):
        if action == "left" or action == "right" or action == "still":
            new_action = {"lateral": action, "longitudinal": 0}
            if action == "still":
                new_action["lateral"] = "central"
            action = new_action
        ini_traj_info = self.traj_info["%.1f" % start_time]

        predict_traj_info = {}

        location = carla.Location(x=ini_traj_info["x_lon"], y=ini_traj_info["x_lat"], z=0.1)

        current_wp = self.carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if action["lateral"] == "left":
            next_wp = current_wp.get_left_lane()
        elif action["lateral"] == "right":
            next_wp = current_wp.get_right_lane()
        else:
            next_wp = current_wp

        if next_wp is not None:
            next_wp = current_wp
        predict_traj_info["road_id"]  = next_wp.road_id
        predict_traj_info["lane_index"] = next_wp.lane_index
        predict_traj_info["x_lat"] = next_wp.transform.location.y
        acceleration = action["longitudinal"]


        predict_traj_info["x_lon"] = ini_traj_info["x_lon"] + ini_traj_info["v_lon"] * time_duration + 0.5 * acceleration * time_duration**2
        predict_traj_info["v_lon"] = ini_traj_info["v_lon"] + acceleration * time_duration
        predict_traj_info["v_lat"] = 0.0
        self.traj_info["%.1f" % (start_time + time_duration)] = predict_traj_info
        self.interpolate(start_time = start_time, end_time = start_time + time_duration)
        return self.traj_info["%.1f" % (start_time + time_duration)]


    def predict_without_action(self, start_time=0.0, time_duration=1.0, is_lane_change=None, action=None):
        ini_traj_info = self.traj_info["%.1f" % start_time]
        if time_duration == 1.0:
            predict_traj_info = {}
            acceleration_dict = {
                "accelerate": 2.0,
                "still": 0.0,
                "brake": -4.0
            }
        road_id, lane_id = ini_traj_info["road_id"], ini_traj_info["lane_id"]
        v_lat = ini_traj_info["v_lat"]
        predict = {}

        # === decide next lane id ===
        if is_lane_change is None:
            if v_lat > 0.5:
                next_lane_id = lane_id + 1
            elif v_lat < -0.5:
                next_lane_id = lane_id - 1
            else:
                next_lane_id = lane_id
        else:
            if is_lane_change and v_lat > 0.5:
                next_lane_id = lane_id + 1
            elif is_lane_change and v_lat < -0.5:
                next_lane_id = lane_id - 1
            else:
                next_lane_id = lane_id

        predict["road_id"] = road_id
        predict["lane_id"] = next_lane_id

        # === get target lane lateral center ===
        try:
            waypoint = self.carla_map.get_waypoint_xodr(road_id, next_lane_id, s=0)
            new_x_lat = waypoint.transform.location.y
        except:
            new_x_lat = ini_traj_info["x_lat"]  # fallback
            predict["lane_id"] = lane_id

        if isclose(v_lat, 0.0):
            estimated_lane_change_time = 100
        else:
            estimated_lane_change_time = round(abs((new_x_lat - ini_traj_info["x_lat"]) / v_lat), 1)

        if not is_lane_change or estimated_lane_change_time > 1:
            acceleration = acceleration_dict.get(action, 0.0)
            predict["x_lat"] = new_x_lat
            predict["x_lon"] = ini_traj_info["x_lon"] + ini_traj_info["v_lon"] * time_duration + 0.5 * acceleration * time_duration ** 2
            predict["v_lon"] = ini_traj_info["v_lon"] + acceleration * time_duration
            predict["v_lat"] = 0
            self.traj_info["%.1f" % (start_time + time_duration)] = predict
            self.interpolate(start_time, start_time + time_duration)
        else:
            predict["x_lat"] = new_x_lat
            predict["x_lon"] = ini_traj_info["x_lon"] + ini_traj_info["v_lon"] * estimated_lane_change_time
            predict["v_lon"] = ini_traj_info["v_lon"]
            predict["v_lat"] = 0
            self.traj_info["%.1f" % (start_time + estimated_lane_change_time)] = predict
            self.predict_without_action(start_time + estimated_lane_change_time, 1 - estimated_lane_change_time,
                                        is_lane_change, action)
            self.interpolate(start_time, start_time + estimated_lane_change_time)
            self.interpolate(start_time + estimated_lane_change_time, start_time + time_duration)

        return self.traj_info["%.1f" % (start_time + time_duration)]

    def interpolate(self, start_time, end_time, time_resolution=0.1):
        if ("%.1f" % start_time) not in self.traj_info or ("%.1f" % end_time) not in self.traj_info:
            raise ValueError("Interpolate between non-existing points")

        ini_info = self.traj_info["%.1f" % start_time]
        end_info = self.traj_info["%.1f" % end_time]

        for t in drange(start_time, end_time, time_resolution):
            if isclose(t, start_time) or isclose(t, end_time):
                continue
            ratio = (end_time - t) / (end_time - start_time)
            interp = {
                "x_lon": ratio * ini_info["x_lon"] + (1 - ratio) * end_info["x_lon"],
                "x_lat": ratio * ini_info["x_lat"] + (1 - ratio) * end_info["x_lat"],
                "v_lon": ratio * ini_info["v_lon"] + (1 - ratio) * end_info["v_lon"],
                "v_lat": ratio * ini_info["v_lat"] + (1 - ratio) * end_info["v_lat"]
            }

            # Optional: use CARLA map to update road_id/lane_id (costly)
            location = carla.Location(x=interp["x_lon"], y=interp["x_lat"], z=0.1)
            waypoint = self.carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
            interp["road_id"] = waypoint.road_id
            interp["lane_id"] = waypoint.lane_id

            self.traj_info["%.1f" % t] = interp



