import math

class TrajUtils:
    veh_length = 5.0
    veh_width = 2.0
    circle_r = 1.3
    tem_len = math.sqrt(circle_r**2 - (veh_width / 2)**2)

    @staticmethod
    def collision_check(traj1, traj2):
        time_series = list(traj1.keys())
        for time in time_series:
            center_list_1 = TrajUtils.get_circle_center_list(traj1[time])
            center_list_2 = TrajUtils.get_circle_center_list(traj2[time])
            for p1 in center_list_1:
                for p2 in center_list_2:
                    dist = TrajUtils.cal_dist(p1, p2)
                    if dist <= 2 * TrajUtils.circle_r:
                        return True
        return False

    @staticmethod
    def get_circle_center_list(traj_point):
        center1 = (traj_point["x_lon"], traj_point["x_lat"])
        if traj_point["v_lon"] == 0:
            heading = 0
        else:
            heading = math.atan(traj_point["v_lat"] / traj_point["v_lon"])
        offset = TrajUtils.veh_length / 2 - TrajUtils.tem_len
        center0 = (
            center1[0] + offset * math.cos(heading),
            center1[1] + offset * math.sin(heading)
        )
        center2 = (
            center1[0] - offset * math.cos(heading),
            center1[1] - offset * math.sin(heading)
        )
        return [center0, center1, center2]

    @staticmethod
    def cal_dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    @staticmethod
    def drange(start, stop, step):
        r = start
        while r < stop:
            yield r
            r += step
