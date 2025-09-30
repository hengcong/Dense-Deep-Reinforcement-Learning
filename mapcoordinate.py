import carla
import csv

# 连接CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

# 加载地图 singlelane400m
world = client.load_world("singlelane400m")
carla_map = world.get_map()

# 生成所有waypoints，间隔1米
waypoints = carla_map.generate_waypoints(1.0)

# 保存到CSV
with open("singlelane400m_waypoints.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "z", "yaw"])
    for wp in waypoints:
        loc = wp.transform.location
        rot = wp.transform.rotation
        writer.writerow([loc.x, loc.y, loc.z, rot.yaw])

print(f"Total waypoints: {len(waypoints)}")
print("Waypoints saved to singlelane400m_waypoints.csv")
