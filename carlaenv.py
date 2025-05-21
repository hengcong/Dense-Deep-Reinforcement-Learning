import carla
import numpy as np
import time
import random
from gym import spaces, core
import sys
import bisect
import conf.conf as conf
import math
import pygame
import os
import json
import copy

from conf.defaultconf import episode
from controller.TreeSearchBVGlobalController import TreeSearchBVGlobalController
from controller.treesearchcarlacontroller import TreeSearchController
from controller.trafficmanagercontroller import TMController
from mtlsp.controller.vehicle_controller import DummyGlobalController
from mtlsp.observation.observation_carla import ObservationCarla
from carla_infoextractor import CarlaInfoExtractor
from envs.vehicle_wrapper import VehicleWrapper
from datetime import datetime

from mtlsp.controller.vehicle_controller.idmcontroller_carla import IDMController

class CarlaEnv(core.Env):
    def __init__(self,num_veh, num_ped, mode="NDE"):
        super(CarlaEnv, self).__init__()
        #self.vehicle_wrapper_dict = None
        self.mode = mode
        self.step_size = 0.05

        # Connect to CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)  # Set the connection timeout
        self.world = self.client.load_world('Town04')  # Load a specific town
        self.map = self.world.get_map()

        self.num_veh = num_veh
        self.num_ped = num_ped

        # initiate ego vehicle, surrounding vehicles, and pedestrians
        self.ego_vehicle = None
        self.ego_vehicle_wrapper = None
        self.vehicles=[] # NOTE: remove later
        self.vehicle_wrapper_list = {}
        self.pedestrians = []
        self.global_controller_instance_list = [
            TreeSearchBVGlobalController(env=self, veh_type="BV"),
            DummyGlobalController(env=self, veh_type="CAV")
        ]

        self.blueprint_library = self.world.get_blueprint_library()

        self.collision_sensor = None
        self.spectator = self.world.get_spectator()

        # traffic manager, control vehicles
        self.traffic_manager = self.client.get_trafficmanager(9200)

        # reinforcement learning definition
        self.action_space = spaces.Box(low=0.001, high=0.999, shape=(1,))
        self.observation_space = spaces.Box(low=-5, high=5, shape=(10,))

        # Spawn points and sensors can be set up here
        self.spawn_points = self.map.get_spawn_points()
        self.lane_map = self._group_spawn_points_by_road_and_lane()

        self.spawn_ego_vehicle()
        self.generate_bv_traffic_flow()
        self.vehicle_map = self._group_vehicles_by_road_and_lane()

        #self.spawn_pedestrians()
        self.experiment_path = "./carla_experiments"
        self.info_extractor = CarlaInfoExtractor(self)
        #self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0  # some customized metric logging

        # pygame setup
        self.camera_sensor = None
        self.latest_image = None
        self.pygame_display_initialized = False
        self.screen = None

        self.setup_sensors()
        # log
        self.collision_happened = False
        self.episode_info = {"id": 0, "start_time": self.get_simulation_time(), "end_time": self.get_simulation_time()}
        self.episode_data = {}
        self.step_data = {}

        # Create a timestamp-based directory name for logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_path = f"./carla_experiments/run_{timestamp}"

        # Ensure the directory exists
        os.makedirs(self.experiment_path, exist_ok=True)

        self.ego_controller = None
        self.bv_controllers = {}

        self.start_location = None
        self.start_time = 0.0

        self.distance_travelled = 0.0
        self.last_location = None


    def _group_spawn_points_by_road_and_lane(self):
        lane_map = {}
        for spawn in self.spawn_points:
            waypoint = self.map.get_waypoint(spawn.location)
            road_id = waypoint.road_id
            lane_id = waypoint.lane_id

            key = (road_id, lane_id)
            if key not in lane_map:
                lane_map[key] = []
            lane_map[key].append(spawn)

        # 🔍 Print the number and positions of spawn points for each (road_id, lane_id)
        # print(f"[INFO] Grouped lane map contains {len(lane_map)} lanes.")
        # for key, spawns in lane_map.items():
        #     print(f"[LANE] road_id={key[0]}, lane_id={key[1]}, num_points={len(spawns)}")
        #     for i, sp in enumerate(spawns):
        #         loc = sp.location
        #         print(f"   └─ [SP-{i}] x={loc.x:.2f}, y={loc.y:.2f}, z={loc.z:.2f}")

        return lane_map

    def _group_vehicles_by_road_and_lane(self):
        vehicle_map = {}
        for vehicle in [self.ego_vehicle]+self.vehicles:
            wp = self.get_waypoint(vehicle)
            key = (wp.road_id, wp.lane_id)
            if key not in vehicle_map:
                vehicle_map[key] = []
            vehicle_map[key].append((vehicle, wp.s))

        return vehicle_map

    def reset(self):
        print("🔁 Resetting CarlaEnv...")
        # Destroy existing actors
        ids_to_destroy = []
        if self.ego_vehicle is not None and self.ego_vehicle.is_alive:
            ids_to_destroy.append(self.ego_vehicle.id)
            print(f"🧹 Destroying ego vehicle {self.ego_vehicle.id}")

        ids_to_destroy += [v.id for v in self.vehicles if v is not None and v.is_alive]
        ids_to_destroy += [p.id for p in self.pedestrians if p is not None and p.is_alive]

        if ids_to_destroy:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in ids_to_destroy])
            print(f"✅ Destroyed {len(ids_to_destroy)} actors.")
        else:
            print("⚠️ No actors to destroy.")

        self.vehicles = []
        self.pedestrians = []
        self.ego_vehicle = None
        self.vehicle_wrapper_list = {}

        self.destroy_sensors()
        self.collision_happened = False

        time.sleep(0.5)
        self.world.tick()

        self.soft_reboot()
        self.vehicle_map = self._group_vehicles_by_road_and_lane()

        self.ego_vehicle_wrapper.update_observation(self)
        for wrapper in self.vehicle_wrapper_list.values():
            wrapper.update_observation(self)

        self.start_location = self.ego_vehicle.get_location()
        self.start_time = self.get_simulation_time()
        self.distance_travelled = 0.0
        self.last_location = None

        self.episode_info["start_time"] = self.get_simulation_time()
        return self.get_state()

    def soft_reboot(self):
        self.spawn_ego_vehicle()
        self.generate_bv_traffic_flow()
        self.spawn_pedestrians()
        self.setup_sensors()

    def spawn_ego_vehicle(self):
        """Spawn an ego vehicle with autopilot or external controller."""
        if self.ego_vehicle:
            self.ego_vehicle.destroy()

        ego_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        spawn_point = random.choice(self.map.get_spawn_points())
        self.ego_vehicle = self.world.spawn_actor(ego_bp, spawn_point)
        self.ego_vehicle_wrapper = VehicleWrapper(self.ego_vehicle)
        self.ego_vehicle_wrapper.set_role("CAV")

        if self.mode == "TM":
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
            self.traffic_manager.ignore_lights_percentage(self.ego_vehicle, 0)
            self.traffic_manager.auto_lane_change(self.ego_vehicle, True)
            self.traffic_manager.vehicle_percentage_speed_difference(self.ego_vehicle, -20)
            self.ego_vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self.ego_controller = TMController(vehicle=self.ego_vehicle)

        elif self.mode == "NDE":
            self.ego_vehicle.set_autopilot(False)
            controller = TreeSearchController(env=self)
            controller.attach_to_vehicle(self.ego_vehicle_wrapper)
            self.ego_vehicle_wrapper.install_controller(controller)
            self.ego_controller = controller

        elif self.mode in ["D2RL", "behavior_policy"]:
            self.ego_vehicle.set_autopilot(False)
            self.ego_controller = None

        # Install controller again if exists
        if self.ego_controller:
            self.ego_vehicle_wrapper.install_controller(self.ego_controller)
            #self.global_controller_instance_list[self.ego_vehicle.id] = self.ego_controller #NOTE: 这有问题

        # self.ego_vehicle_wrapper.update_observation(self)

    def spawn_background_vehicle(self, spawn_point, speed, road_id, lane_id):
        """ Generate background vehicles in CARLA"""
        allowed_brands = [
            "audi.tt",
            "bmw.grandtourer",
            "chevrolet.impala",
            "citroen.c3",
            "jeep.wrangler_rubicon",
            "lincoln.mkz_2020",
            "mini.cooper_s",
            "nissan.micra",
            "seat.leon",
            "tesla.model3",
            "toyota.prius",
            "volkswagen.t2",
            "mercedes.coupe"
        ]

        vehicle_blueprints = [
            self.blueprint_library.find(f"vehicle.{brand}") for brand in allowed_brands
        ]
        vehicle_bp = random.choice(vehicle_blueprints)
        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle:
            wrapper = VehicleWrapper(vehicle)
            self.vehicles.append(vehicle)
            self.vehicle_wrapper_list[vehicle.id] = wrapper
            wrapper.set_role("BV")

            if self.mode == "TM":
                vehicle.set_autopilot(True, self.traffic_manager.get_port())

                # Random parameters
                self.traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(1.0, 3.5))
                self.traffic_manager.ignore_lights_percentage(vehicle, random.randint(0, 30))
                self.traffic_manager.auto_lane_change(vehicle, random.choice([True, False]))
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-20, 10))
                controller = TMController(vehicle=vehicle)
                """    
                # Choose driving style randomly
                style = random.choice(["aggressive", "normal", "cautious"])

                if style == "aggressive":
                    self.traffic_manager.set_distance_to_leading_vehicle(vehicle, 1.0)
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -20)
                    self.traffic_manager.auto_lane_change(vehicle, True)

                elif style == "normal":
                    self.traffic_manager.set_distance_to_leading_vehicle(vehicle, 2.0)
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 0)
                    self.traffic_manager.auto_lane_change(vehicle, random.choice([True, False]))

                elif style == "cautious":
                    self.traffic_manager.set_distance_to_leading_vehicle(vehicle, 4.0)
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, 10)
                    self.traffic_manager.auto_lane_change(vehicle, False)
                """
            elif self.mode == "NDE":
                vehicle.set_autopilot(False)
                controller = TreeSearchController(env= self)
                controller.attach_to_vehicle(wrapper) #NOTE: 这个逻辑需要盘一下
                wrapper.install_controller(controller)
                #wrapper.update_observation(self)

                if wrapper.controller:
                    print(f"[✓] Controller attached to vehicle {vehicle.id}")
                else:
                    print(f"[✗] Controller missing for vehicle {vehicle.id}")
                    print("💡 Current wrapper list keys:", list(self.vehicle_wrapper_list.keys()))

            elif self.mode == "D2RL":
                vehicle.set_autopilot(False)
                controller = None  # Replace with your RLController

            elif self.mode == "behavior_policy":
                vehicle.set_autopilot(False)
                controller = None  # Replace with your behavior policy controller

            else:
                vehicle.set_autopilot(False)
                controller = None

            # if controller is not None:
            #     self.global_controller_instance_list[vehicle.id] = controller
            return wrapper
        return None

    def spawn_pedestrians(self):
        """Spawns pedestrians in the CARLA environment and assigns AI controllers."""

        # Remove existing pedestrians before spawning new ones
        for pedestrian in self.pedestrians:
            if pedestrian.is_alive:
                pedestrian.destroy()
        self.pedestrians = []

        # Get the pedestrian blueprint
        pedestrian_bp = random.choice(self.blueprint_library.filter("walker.pedestrian.*"))
        controller_bp = self.blueprint_library.find("controller.ai.walker")

        # Retrieve spawn points for pedestrians
        spawn_points = []
        for _ in range(self.num_ped):
            spawn_point = carla.Transform()
            spawn_point.location = self.world.get_random_location_from_navigation()
            if spawn_point.location:
                spawn_points.append(spawn_point)

        # Spawn pedestrians
        for spawn_point in spawn_points:
            pedestrian = self.world.try_spawn_actor(pedestrian_bp, spawn_point)
            if pedestrian:
                self.pedestrians.append(pedestrian)

                # Assign AI controller to the pedestrian
                controller = self.world.spawn_actor(controller_bp, carla.Transform(), attach_to=pedestrian)
                if controller:
                    # Start AI-controlled movement
                    controller.start()
                    controller.go_to_location(self.world.get_random_location_from_navigation())
                    controller.set_max_speed(random.uniform(0.5, 1.5))  # Set random walking speed

    def generate_bv_traffic_flow(self):
        all_spawn_lanes = [(key, sps) for key, sps in self.lane_map.items() if len(sps) > 1]
        random.shuffle(all_spawn_lanes)

        total_needed = self.num_veh
        spawned = 0
        vehicle_records = []
        used_positions = set()

        for (road_id, lane_id), spawn_points in all_spawn_lanes:
            if spawned >= total_needed:
                break

            spawn_points = sorted(spawn_points, key=lambda sp: sp.location.x)
            max_in_lane = min(len(spawn_points), total_needed - spawned)
            previous = None

            for i in range(max_in_lane):
                sp = spawn_points[i]
                pos = round(sp.location.x, 1)

                if pos in used_positions:
                    continue

                if previous is None:
                    speed = self.generate_random_speed()
                    mode = "FF"
                else:
                    if random.random() < conf.CF_percent:
                        speed, _ = self.generate_CF_vehicle(previous)
                        mode = "CF"
                    else:
                        speed = self.generate_random_speed()
                        mode = "FF"

                wrapper = self.spawn_background_vehicle(sp, speed, road_id, lane_id)
                veh = wrapper.vehicle if wrapper else None
                if veh:
                    used_positions.add(pos)
                    vehicle_records.append((veh.id, pos, mode))
                    previous = {"speed": speed, "position": pos}
                    spawned += 1

                if spawned >= total_needed:
                    break

        print("=== Background Vehicles ===")
        for vid, pos, mode in vehicle_records:
            print(f"[{mode}] Vehicle ID: {vid}, Position X: {pos:.1f}")

    def sample_CF_FF_mode(self):
        """Randomly choose the Cf or FF mode to generate vehicles.

        Returns:
            str: Mode ID.
        """
        random_number_CF = np.random.uniform()
        if random_number_CF > conf.CF_percent:
            return "FF"
        else:
            return "CF"

    def generate_FF_vehicle(self):
        """Generate a Free-Flow (FF) vehicle with independent speed and spawn point."""
        spawn_point = random.choice(self.spawn_points)
        speed = self.generate_random_speed()
        return speed, spawn_point.location.x

    def generate_CF_vehicle(self, front_speed_position):
        """Generate a Car-Following (CF) vehicle based on the preceding vehicle's speed and position."""
        prev_speed = front_speed_position["speed"]
        prev_position = front_speed_position["position"]

        # Ensure a safe distance
        min_gap = 5  # Minimum gap in meters
        max_gap = 20  # Maximum gap in meters
        position = max(prev_position - np.random.uniform(min_gap, max_gap), prev_position - 50)  # 保证CF车辆不会超车

        # Reduce speed slightly to simulate following behavior
        speed = max(prev_speed - np.random.uniform(0, 5), 0)
        return speed, position

    def generate_random_speed(self):
        """Generate a random speed for a vehicle based on the NDD distribution."""
        random_number = np.random.uniform()
        idx = bisect.bisect_left(conf.speed_CDF, random_number)
        return conf.v_to_idx_dic.inverse[idx]

    def setup_sensors(self):
        # Collision sensor
        if self.collision_sensor is None or not self.collision_sensor.is_alive:
            sensor_bp = self.blueprint_library.find('sensor.other.collision')
            sensor_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
            self.collision_sensor = self.world.spawn_actor(sensor_bp, sensor_transform, attach_to=self.ego_vehicle)
            self.collision_sensor.listen(lambda event: self._on_collision(event))

        # RGB camera sensor
        if self.camera_sensor is None or not self.camera_sensor.is_alive:
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')

            camera_transform = carla.Transform(
                carla.Location(x=-6.5, y=0, z=2.5),
                carla.Rotation(pitch=-15, yaw=0, roll=0)
            )

            self.camera_sensor = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.ego_vehicle)
            self.camera_sensor.listen(lambda image: self._process_camera_image(image))
        print("[✓] camera_sensor set up:", self.camera_sensor.is_alive)

    def destroy_sensors(self):
        if self.camera_sensor is not None:
            if self.camera_sensor.is_alive:
                try:
                    self.camera_sensor.stop()
                except:
                    pass
                try:
                    self.camera_sensor.destroy()
                except:
                    pass
            self.camera_sensor = None

        if self.collision_sensor is not None:
            if self.collision_sensor.is_alive:
                try:
                    self.collision_sensor.stop()
                except:
                    pass
                try:
                    self.collision_sensor.destroy()
                except:
                    pass
            self.collision_sensor = None

    def render_image(self):
        if self.latest_image is None:
            return

        if not self.pygame_display_initialized:
            pygame.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 24)
            self.screen = pygame.display.set_mode((self.latest_image.shape[1], self.latest_image.shape[0]))
            pygame.display.set_caption("Ego Camera View")
            self.pygame_display_initialized = True
            self._last_time = time.time()
            self._fps = 0.0

        current_time = time.time()
        dt = current_time - self._last_time
        if dt > 0:
            self._fps = 1.0 / dt
        self._last_time = current_time

        surface = pygame.surfarray.make_surface(self.latest_image.swapaxes(0, 1))
        self.screen.blit(surface, (0, 0))

        fps_text = self.font.render(f"FPS: {self._fps:.2f}", True, (255, 255, 0))
        self.screen.blit(fps_text, (10, 10))

        # === Abstract bird's-eye map ===
        try:
            mini_map_size = 200
            scale = 1.0  # pixels per meter
            padding = 10
            center = (mini_map_size // 2, mini_map_size // 2)

            mini_surface = pygame.Surface((mini_map_size, mini_map_size))
            mini_surface.fill((40, 40, 40))

            ego_loc = self.ego_vehicle.get_location()

            # Draw simplified lane graph using walkable waypoints
            for waypoint in self.map.generate_waypoints(2.0):
                next_wp_list = waypoint.next(2.0)
                if not next_wp_list:
                    continue
                next_wp = next_wp_list[0]

                x1 = int(center[0] - (waypoint.transform.location.x - ego_loc.x) * scale)
                y1 = int(center[1] - (waypoint.transform.location.y - ego_loc.y) * scale)
                x2 = int(center[0] - (next_wp.transform.location.x - ego_loc.x) * scale)
                y2 = int(center[1] - (next_wp.transform.location.y - ego_loc.y) * scale)

                pygame.draw.line(mini_surface, (90, 90, 90), (x1, y1), (x2, y2), 1)

            # Draw vehicles
            for vehicle in [self.ego_vehicle] + self.vehicles:
                loc = vehicle.get_location()
                dx = (loc.x - ego_loc.x) * scale
                dy = (loc.y - ego_loc.y) * scale
                vx = int(center[0] - dx)
                vy = int(center[1] - dy)
                color = (0, 255, 0) if vehicle.id == self.ego_vehicle.id else (200, 200, 200)
                pygame.draw.circle(mini_surface, color, (vx, vy), 4)

            # Draw traffic lights
            for tl in self.world.get_actors().filter("traffic.traffic_light*"):
                loc = tl.get_location()
                dx = (loc.x - ego_loc.x) * scale
                dy = (loc.y - ego_loc.y) * scale
                tx = int(center[0] - dx)
                ty = int(center[1] - dy)
                color = {
                    carla.TrafficLightState.Red: (255, 0, 0),
                    carla.TrafficLightState.Yellow: (255, 255, 0),
                    carla.TrafficLightState.Green: (0, 255, 0),
                    carla.TrafficLightState.Off: (100, 100, 100)
                }.get(tl.state, (50, 50, 50))
                pygame.draw.circle(mini_surface, color, (tx, ty), 3)

            # Border
            pygame.draw.rect(mini_surface, (200, 200, 0), (0, 0, mini_map_size, mini_map_size), 2)

            # Blit to bottom-right
            self.screen.blit(mini_surface, (
                self.latest_image.shape[1] - mini_map_size - padding,
                self.latest_image.shape[0] - mini_map_size - padding
            ))

        except Exception as e:
            print(f"[⚠️ Abstract mini-map failed] {e}")

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def _process_camera_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        self.latest_image = array

    def _on_collision(self, event):
        self.collision_happened = True

    def check_collision(self):
        return self.collision_happened

    def get_state(self):
        obs = ObservationCarla(veh_id=self.ego_vehicle.id, time_stamp=self.get_simulation_time())
        obs.update(env=self)
        return obs.information

    def get_av_obs(self):
        cav = self.ego_vehicle_wrapper

        if cav is None:
            print("[!] No CAV wrapper found.")
            return {}

        if cav.observation is None:
            print(f"[!] CAV observation is None.")
            return {}

        cav_observation = cav.observation.information
        return_information = copy.deepcopy(cav_observation)

        success_count, fail_count = 0, 0

        for key in return_information:
            info = return_information[key]
            if info and info["veh_id"] != self.ego_vehicle.id:
                veh_id = int(info["veh_id"])  # Ensure ID is int
                print(f"[🔍] Processing vehicle ID: {veh_id}")

                wrapper = self.vehicle_wrapper_list.get(veh_id)
                if wrapper is None:
                    print(f"[!] Vehicle ID {veh_id} not found in vehicle_wrapper_list.")
                    fail_count += 1
                    continue

                if wrapper.controller is None:
                    print(f"[!] Controller not found for vehicle {veh_id}")
                    fail_count += 1
                    continue

                try:
                    ndd_pdf = np.array(wrapper.controller.ndd_pdf)
                    return_information[key]["ndd_pdf"] = ndd_pdf.tolist()
                    print(f"[✓] Added NDD PDF for vehicle {veh_id}")
                    success_count += 1
                except Exception as e:
                    print(f"[✗] Failed to get NDD PDF for vehicle {veh_id}: {e}")
                    fail_count += 1

        print(f"[✔️] get_av_obs summary: {success_count} success, {fail_count} failed.")
        return return_information

    def get_surrounding_vehicles(self, target_vehicle=None):
        """
        Find the closest 6 surrounding vehicles (Lead, Foll, LeftLead, LeftFoll, RightLead, RightFoll)
        within 120m based on s-distance and heading consistency.
        """
        if target_vehicle is None:
            target_vehicle = self.ego_vehicle

        target_wp = self.get_waypoint(target_vehicle)
        target_s = target_wp.s
        target_lane_id = target_wp.lane_id
        target_road_id = target_wp.road_id
        target_yaw = target_vehicle.get_transform().rotation.yaw
        ref_yaw = target_wp.transform.rotation.yaw

        ego_s_adj = get_adjusted_s(target_s, target_yaw, ref_yaw)

        surrounding_vehicles = {
            'Lead': None, 'Foll': None,
            'LeftLead': None, 'LeftFoll': None,
            'RightLead': None, 'RightFoll': None
        }
        min_distances = {key: float('inf') for key in surrounding_vehicles}

        # Define lane search targets
        search_lanes = {
            'Lead': (target_road_id, target_lane_id),
            'Foll': (target_road_id, target_lane_id)
        }

        left_wp = target_wp.get_left_lane()
        if left_wp and left_wp.lane_id * target_lane_id > 0:
            search_lanes['LeftLead'] = (left_wp.road_id, left_wp.lane_id)
            search_lanes['LeftFoll'] = (left_wp.road_id, left_wp.lane_id)

        right_wp = target_wp.get_right_lane()
        if right_wp and right_wp.lane_id * target_lane_id > 0:
            search_lanes['RightLead'] = (right_wp.road_id, right_wp.lane_id)
            search_lanes['RightFoll'] = (right_wp.road_id, right_wp.lane_id)

        for label in search_lanes:
            road_id, lane_id = search_lanes[label]
            lane_vehicles = self.vehicle_map.get((road_id, lane_id), [])

            for vehicle, s in lane_vehicles:
                if vehicle.id == target_vehicle.id:
                    continue

                wp = self.get_waypoint(vehicle)
                veh_yaw = vehicle.get_transform().rotation.yaw
                veh_ref_yaw = wp.transform.rotation.yaw
                veh_s_adj = get_adjusted_s(wp.s, veh_yaw, veh_ref_yaw)

                delta_s = veh_s_adj - ego_s_adj

                if "Lead" in label and delta_s > 0:
                    if delta_s < min_distances[label] and delta_s <= 120:
                        surrounding_vehicles[label] = vehicle
                        min_distances[label] = delta_s
                elif "Foll" in label and delta_s < 0:
                    if abs(delta_s) < min_distances[label] and abs(delta_s) <= 120:
                        surrounding_vehicles[label] = vehicle
                        min_distances[label] = abs(delta_s)

        return surrounding_vehicles

    def get_relative_position(self,ego_vehicle, surrounding_vehicle):
        ego_wp = self.get_waypoint(ego_vehicle)
        surrounding_vehicle_wp = self.get_waypoint(surrounding_vehicle)
        longitudinal_offset = surrounding_vehicle_wp.s -ego_wp.s
        lateral_offset = surrounding_vehicle_wp.lane_id - ego_wp.lane_id

        return longitudinal_offset, lateral_offset

    def step(self):
        self.vehicle_map = self._group_vehicles_by_road_and_lane()
        self.step_data = {}

        control_info_list = []
        for global_controller in self.global_controller_instance_list:
            result = global_controller.step()  # TreeSearchNADEBackgroundController.step()
            control_info_list.append(result)

        current_time = self.get_simulation_time()
        self.episode_data[current_time] = self.step_data

        current_location = self.ego_vehicle.get_location()
        if self.last_location is not None:
            step_distance = self.get_distance(self.last_location, current_location)
            self.distance_travelled += step_distance
        self.last_location = current_location

        self.world.tick()
        self.episode_info["end_time"] = self.get_simulation_time()
        self.render_image()

        self.info_extractor.get_snapshot_info(control_info_list)


    def compute_reward(self):
        reward = 0.0

        # Example: Penalize collisions
        if self.check_collision():
            reward -= 10

        # Example: Reward for staying within a target speed range
        velocity = self.ego_vehicle.get_velocity().length()
        if 10 < velocity < 30:  # Ideal speed range
            reward += 1

        return reward

    def check_done(self):
        reason = None
        stop = False
        additional_info = {}

        if self.check_collision():
            reason = {1: "CAV and BV collision"}
            stop = True
            additional_info = {}

        elif self.ego_vehicle is None:
            reason = {2: "CAV leaves network"}
            stop = True

        elif len(self.get_vehicle_list()) == 0:
            reason = {3: "All vehicles leave network"}
            stop = True

        elif self.distance_travelled > 800.0:
            reason = {4: "CAV traveled over 800m"}
            stop = True

        elif self.get_simulation_time() - self.start_time > 60.0:
            reason = {5: "Timeout: over 60s elapsed"}
            stop = True

        if stop:
            if not hasattr(self, "episode_info"):
                self.episode_info = {}
            self.episode_info["end_time"] = self.get_simulation_time()
            self.info_extractor.get_terminate_info(stop=True, reason=reason, additional_info=additional_info)
            print("[✓] Calling get_terminate_info...")

        return stop, reason, additional_info

    def get_available_lanes(self):
        """get available road and lane"""
        topology = self.map.get_topology()  # 获取 CARLA 地图拓扑
        lane_dict = {}

        for segment in topology:
            start_wp, end_wp = segment
            lane_id = start_wp.lane_id
            road_id = start_wp.road_id

            if (road_id, lane_id) not in lane_dict:
                lane_dict[(road_id, lane_id)] = start_wp
        return lane_dict

    def log_episode(self):
        episode_id = self.episode_info.get("id", 0)
        start_time = self.episode_info.get("start_time", 0)
        end_time = self.episode_info.get("end_time", 0)
        duration = end_time - start_time

        total_reward = 0.0
        collision_flag = self.collision_happened

        for step_time, step_data in self.episode_data.items():
            ego_step = step_data.get(self.ego_vehicle.id)
            if ego_step:
                total_reward += self.compute_reward()
            if self.collision_happened:
                collision_flag = True

        episode_summary = {
            "episode_id": episode_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "total_reward": total_reward,
            "collision": collision_flag,
            "step_data": self.episode_data,
        }

        if not hasattr(self, "episode_logs"):
            self.episode_logs = []
        self.episode_logs.append(episode_summary)

        # update counter
        if not hasattr(self, "crash_count"):
            self.crash_count = 0
        if not hasattr(self, "total_episode_run"):
            self.total_episode_run = 0
        if not hasattr(self, "worker_id"):
            self.worker_id = 0

        self.total_episode_run += 1
        if collision_flag:
            self.crash_count += 1

        # save every 50 episodes
        if self.episode_info["id"] % 50 == 0:
            np.save(os.path.join(self.experiment_path, f"weight{self.worker_id}.npy"),
                    np.array([self.crash_count, self.total_episode_run, self.episode_info["id"]]))

        # === save raw data===
        raw_data_path = os.path.join(self.experiment_path, "raw_data")
        os.makedirs(raw_data_path, exist_ok=True)
        raw_episode_path = os.path.join(raw_data_path, f"episode_{episode_id:04d}.json")

        saved_step_data = {
            str(step_time): {
                str(veh_id): {
                    "observation": data["observation"],
                    "action": data["action"]
                } for veh_id, data in step_data.items()
            } for step_time, step_data in self.episode_data.items()
        }

        raw_record = {
            "episode_id": episode_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "total_reward": total_reward,
            "collision": collision_flag,
            "step_data": saved_step_data
        }

        with open(raw_episode_path, "w") as f:
            json.dump(raw_record, f, indent=2)
        print(f"📁 Saved raw episode data to {raw_episode_path}")

        self.episode_data = {}
        self.step_data = {}
        self.episode_info["id"] += 1

    def get_simulation_time(self):
        return self.world.get_snapshot().timestamp.elapsed_seconds

    def get_vehicle_position(self, vehicle):
        """get vehicle position"""
        return vehicle.get_location().x

    # TODO: Due to CARLA's built-in physics, direct lateral speed constraints are unnecessary.
    def set_vehicle_max_lateralspeed(self, vehicle, max_steer=0.3):
        
        control = vehicle.get_control()
        control.steer = max(-max_steer, min(max_steer, control.steer))
        vehicle.apply_control(control)

    def get_av_ttc(self):
        obs = ObservationCarla(veh_id=self.ego_vehicle.id, time_stamp=self.get_simulation_time())
        obs.update(env=self)
        observation = obs.information

        lead_obs = observation.get("Lead")
        follow_obs = observation.get("Foll")
        ego_obs = observation.get("Ego")

        distance_front, ttc_front = float('inf'), float('inf')
        distance_back, ttc_back = float('inf'), float('inf')

        if lead_obs is not None:
            distance_front, ttc_front = self.get_ttc(lead_obs, ego_obs)

        if follow_obs is not None:
            distance_back, ttc_back = self.get_ttc(ego_obs, follow_obs)

        min_distance = min(distance_front, distance_back)
        min_ttc = min(ttc_front, ttc_back)

        # === 替代非法数值为 10000 ===
        if math.isinf(min_distance) or math.isnan(min_distance):
            min_distance = 10000.0
        if math.isinf(min_ttc) or math.isnan(min_ttc):
            min_ttc = 10000.0

        return min_distance, min_ttc

    def get_ttc(self, lead_obs, follow_obs):
        lead_pos = self.get_waypoint_from_position(lead_obs["position"]).s
        follow_pos = self.get_waypoint_from_position(follow_obs["position"]).s
        lead_vel = lead_obs["speed"]
        follow_vel = follow_obs["speed"]

        distance = lead_pos - follow_pos - 5.0  # 5 meters as assumed vehicle length
        relative_speed = follow_vel - lead_vel

        if relative_speed <= 0:
            return distance, float('inf')
        else:
            ttc = distance / relative_speed
            return distance, min(ttc, 10000)

    def get_waypoint_from_position(self, position):
        loc = carla.Location(x=position[0], y=position[1], z=0.0)
        return self.map.get_waypoint(loc)

    def add_background_vehicles(self, vlist, add_to_vlist=True, add_to_carla=True):
        pass

    def generate_traffic_flow(self, init_info=None):
        pass

    def track_ego_vehicle(self):
        """Make the CARLA spectator camera follow the ego vehicle."""
        if not self.ego_vehicle:
            return

        ego_transform = self.ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_rotation = ego_transform.rotation

        # Set relative position (behind and above)
        offset_back = 6.5  # meters behind the vehicle
        offset_up = 2.5  # meters above

        # Convert ego yaw to radians
        yaw_rad = math.radians(ego_rotation.yaw)

        # Compute offset location behind the ego vehicle
        dx = -offset_back * math.cos(yaw_rad)
        dy = -offset_back * math.sin(yaw_rad)
        dz = offset_up

        camera_location = carla.Location(
            x=ego_location.x + dx,
            y=ego_location.y + dy,
            z=ego_location.z + dz
        )

        camera_rotation = carla.Rotation(
            pitch=-10.0,
            yaw=ego_rotation.yaw,
            roll=0.0
        )

        self.spectator.set_transform(carla.Transform(camera_location, camera_rotation))

    def get_vehicle_list(self):
        return [actor.id for actor in self.world.get_actors() if 'vehicle' in actor.type_id]

    def get_distance(self, location1, location2):
        dx = location1.x - location2.x
        dy = location1.y - location2.y
        dz = location1.z - location2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def get_waypoint(self, vehicle):
        return self.map.get_waypoint(vehicle.get_location())

    def get_vehicle_by_id(self, vid):
        for v in self.vehicles:
            if v.id == vid:
                return v
        return None

    def get_controller_by_id(self, veh_id):
        wrapper = self.vehicle_wrapper_list.get(veh_id)
        if wrapper:
            return wrapper.controller
        return None

def get_adjusted_s(s, yaw, ref_yaw, threshold_deg=90):
    angle_diff = abs((yaw - ref_yaw + 180) % 360 - 180)
    return s if angle_diff < threshold_deg else -s


def main():
    env = CarlaEnv(num_veh=20, num_ped=10)


if __name__=="__main__":
    main()

