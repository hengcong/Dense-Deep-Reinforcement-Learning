from mtlsp.observation.observation_carla import ObservationCarla
from typing import Optional, Dict, Any
import numpy as np
import carla
class VehicleWrapper:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.controller = None
        self.controlled_flag = False
        self.observation = None  # Observation object if needed
        self.id = vehicle.id     # Alias for convenience
        self.role = None         # Optional: 'CAV' or 'BV' or 'Pedestrian'
        self.controlled_duration = 0
        self.simulate_physics_enabled = True
        self.cached_transform = None
        self.cached_velocity = None

        # Non-physics step size (seconds). Keep aligned with your env dt.
        self.kinematic_dt = 0.05

        # Warmup config
        self.in_warmup = False
        self.warmup_steps_remaining = 0

    def install_controller(self, controller):
        self.controller = controller
        self.vehicle.set_autopilot(False)
        if hasattr(self.controller, 'attach_to_vehicle'):
            controller.attach_to_vehicle(self)

    def reset_control_state(self):
        if self.controller and hasattr(self.controller, 'reset'):
            self.controller.reset()

    def is_action_legal(self, env, action):
        """
        Check whether the action is legal for this vehicle in the current CARLA lane topology.

        Args:
            action (dict): Action with keys "lateral" and "longitudinal", e.g., {"lateral": "left", "longitudinal": 0.5}

        Returns:
            bool: True if legal, False otherwise.
        """
        if "lateral" not in action:
            return False  # action missing key

        lateral_cmd = action["lateral"]

        # +1: check left lane, -1: check right lane
        if lateral_cmd == "left":
            return env.has_adjacent_lane(self.vehicle, direction=-1)
        elif lateral_cmd == "right":
            return env.has_adjacent_lane(self.vehicle, direction=+1)
        else:
            return True  # "keep" or unknown → assume legal

    def step(self):
        if self.controller and hasattr(self.controller, 'step'):
            self.controller.step()

    def update(self, env, dt:Optional[float]=None):
        if isinstance(dt, (int, float)) and dt >0:
            self.kinematic_dt = float(dt)
        else:
            fds = env.world.get_settings().fixed_delta_seconds
            if isinstance(fds, (int, float)) and fds > 0:
                self.kinematic_dt = float(fds)
        dt = self.kinematic_dt
        action = self.controller.action

        if self.simulate_physics_enabled:
            self.update_physics(action)
        else:
            self.update_kinematic(action, dt)

    def update_physics(self, action):
        acc = float(action.get("longitudinal", 0.0))
        if acc > 0.0:
            throttle = min(max(acc / 3.0, 0.0), 1.0) + 0.4
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(max(-acc / 8.0, 0.0), 1.0)

        lat = action.get("lateral", "central")
        if isinstance(lat, (int, float)):
            steer = float(lat)
        else:
            t = str(lat).lower()
            if t in ("left", "l"):
                steer = -0.3
            elif t in ("right", "r"):
                steer = 0.3
            else:
                steer = 0.0
        steer = max(min(steer, 1.0), -1.0)

        ctrl = carla.VehicleControl(
            throttle=float(min(max(throttle, 0.0), 1.0)),
            brake=float(min(max(brake, 0.0), 1.0)),
            steer=float(steer),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(ctrl)

    def update_kinematic(self, action, dt):
        if self.cached_transform is None:
            self.cached_transform = self.vehicle.get_transform()
        if self.cached_velocity is None:
            v = self.vehicle.get_velocity()
            self.cached_velocity = type(v)(v.x, v.y, v.z)

        acc = float(action.get("longitudinal", 0.0))
        val = action.get("lateral", 0.0)
        if isinstance(val, (int, float)):
            steer = float(val)
        else:
            t = str(val).lower()
            if t in ("left", "l"):
                steer = -0.3
            elif t in ("right", "r"):
                steer = 0.3
            elif t in ("straight", "center", "centre", "central", "c", "s"):
                steer = 0.0
            else:
                steer = 0.0
        # steer = float(action.get("lateral", 0.0))
        steer = max(-1.0, min(1.0, steer))

        yaw = np.radians(self.cached_transform.rotation.yaw)
        v = self.cached_velocity
        v_f = v.x * np.cos(yaw) + v.y * np.sin(yaw)
        v_f_new = max(0.0, v_f + acc *dt)
        vx = v_f_new *np.cos(yaw)
        vy = v_f_new *np.sin(yaw)

        x = self.cached_transform.location.x + vx*dt
        y = self.cached_transform.location.y + vy*dt

        yaw_deg = self.cached_transform.rotation.yaw + steer * 0.6 * dt * 180 / np.pi
        # if self.vehicle.id == 780:
        #     print(f"Vehicle {self.vehicle.id} cached_transform: {self.cached_transform}")
        loc = carla.Location(x, y, self.cached_transform.location.z)
        rot = carla.Rotation(self.cached_transform.rotation.pitch, yaw_deg, self.cached_transform.rotation.roll)
        new_tf = carla.Transform(loc, rot)

        self.vehicle.set_transform(new_tf)
        self.vehicle.set_target_velocity(carla.Vector3D(vx, vy, 0.0))

        self.cached_transform = new_tf
        self.cached_velocity = carla.Vector3D(vx, vy, 0.0)

    # def update(self, env, dt=0.05):
    #     # if not self.controller or self.controller.action is None:
    #     #     return
    #     # action = self.controller.action
    #     # if not self.is_action_legal(env, action):
    #     #     return
    #     #
    #     # acc = float(action.get("longitudinal", 0.0))
    #     # steer_cmd = str(action.get("lateral", "central"))
    #     #
    #     # if self.cached_transform is None:
    #     #     try:
    #     #         self.cached_transform= self. vehicle.get_transform()
    #     #     except Exception:
    #     #         return
    #     #
    #     # if self.cached_velocity is None:
    #     #     try:
    #     #         v =self.vehicle.get_velocity()
    #     #         self.cached_velocity = carla.Vector3D(v.x, v.y, v.z)
    #     #     except Exception:
    #     #         self.cached_velocity = carla.Vector3D(0.0, 0.0, 0.0)
    #     #
    #     # if self.simulate_physics_enabled:
    #     #     self._apply_physics_control(acc, steer_cmd)
    #     #
    #     # else:
    #     #     transform = self.cached_transform
    #     #     velocity = self.cached_velocity
    #     #
    #     #     location = transform.location
    #     #     yaw = transform.rotation.yaw
    #     #     yaw_rad = np.radians(yaw)
    #     #
    #     #     vx, vy = velocity.x, velocity.y
    #     #     speed = float(np.hypot(vx, vy))
    #     #     speed = max(0.0, speed + acc * dt)
    #     #
    #     #     if hasattr(self, "max_speed"):
    #     #         speed = min(speed, float(self.max_speed))
    #     #
    #     #     lateral_map = {"left":0.3, "right":-0.3, "central":0.0}
    #     #     lateral_shift =float(lateral_map[steer_cmd, 0.0])
    #     #     dx = speed * dt * np.cos(yaw_rad) - lateral_shift * np.sin(yaw_rad)
    #     #     dy = speed * dt * np.sin(yaw_rad) + lateral_shift * np.cos(yaw_rad)
    #     #
    #     #     new_x = location.x + dx
    #     #     new_y = location.y + dy
    #     #     ground_z = self._ground_z_vehicle(env, new_x, new_y, location.z)
    #     #     z_offset = getattr(self, "ground_offset", 0.05)
    #     #     new_location = carla.Location(x=new_x, y=new_y, z=ground_z + z_offset)
    #     #
    #     #     # Keep yaw unchanged during lateral shift emulation
    #     #     new_rotation = carla.Rotation(yaw=yaw, pitch=0.0, roll=0.0)
    #     #
    #     #     self.cached_transform = carla.Transform(new_location, new_rotation)
    #     #     self.cached_velocity = carla.Vector3D(x=speed * np.cos(yaw_rad),
    #     #                                           y=speed * np.sin(yaw_rad),
    #     #                                           z=0.0)
    #     #
    #     # if steer_cmd in ("left", "right"):
    #     #     self.controlled_duration += 1
    #     # self.controlled_flag = True
    #
    #     if self.controller and self.controller.action is not None:
    #         action = self.controller.action
    #         if self.is_action_legal(env, action):
    #             acc = action["longitudinal"]
    #             steer_cmd = action["lateral"]
    #             #print(f"[DEBUG] action: {action}")
    #             transform = self.cached_transform
    #             velocity = self.cached_velocity
    #
    #             if transform is None or velocity is None:
    #                 print(f"[ERROR] vehicle.{self.vehicle.id} missing cached transform or velocity, skipping update.")
    #                 return
    #             location = transform.location
    #             yaw = transform.rotation.yaw
    #
    #             vx, vy = velocity.x, velocity.y
    #             speed = np.sqrt(vx ** 2 + vy ** 2)
    #
    #             speed += acc * dt
    #             speed = max(speed, 0.0)
    #
    #             yaw_rad = np.radians(yaw)
    #
    #             lateral_map = {"left": 0.3, "right": -0.3, "central": 0.0}
    #             lateral_shift = lateral_map.get(steer_cmd, 0.0)
    #
    #             dx = speed * dt * np.cos(yaw_rad) - lateral_shift * np.sin(yaw_rad)
    #             dy = speed * dt * np.sin(yaw_rad) + lateral_shift * np.cos(yaw_rad)
    #             new_location = carla.Location(x=location.x + dx, y=location.y + dy, z=location.z)
    #
    #             new_rotation = carla.Rotation(yaw=yaw, pitch=0.0, roll=0.0)
    #
    #             self.cached_transform = carla.Transform(new_location, new_rotation)
    #             self.cached_velocity = carla.Vector3D(x=speed * np.cos(yaw_rad),
    #                                                   y=speed * np.sin(yaw_rad),
    #                                                   z=0.0)
    #
    #             if steer_cmd in ["left", "right"]:
    #                 self.controlled_duration += 1
    #             self.controlled_flag = True
    #
    #             # physic mode
    #             # control = carla.VehicleControl()
    #             # control.throttle = np.clip(acc / 3.0, 0, 1)+0.4 if acc > 0 else 0
    #             # control.brake = np.clip(-acc / 8.0, 0, 1) if acc <= 0 else 0
    #             # control.steer = {"left": -0.3, "right": 0.3, "central": 0.0}[steer_cmd]
    #             # self.vehicle.apply_control(control)
    #             #print(
    #             #    f"[DEBUG] Vehicle {self.id} location: {self.vehicle.get_location()}, velocity: {self.vehicle.get_velocity()}")
    #             #print(f"[DEBUG] Vehicle {self.id} control => Throttle: {control.throttle:.2f}, "
    #                   #f"Brake: {control.brake:.2f}, Steer: {control.steer:.2f}")
    #
    #
    #             # update flags
    #             if steer_cmd in ["left", "right"]:
    #                 self.controlled_duration += 1
    #             self.controlled_flag = True

    def set_role(self, role_name):
        self.role = role_name

    def set_observation(self, obs):
        self.observation = obs

    def update_observation(self, env, time_stamp=None):
        if time_stamp is None:
            time_stamp = env.get_simulation_time()
        obs = ObservationCarla(veh_id=self.vehicle.id, time_stamp=time_stamp)
        obs.update(env)
        self.set_observation(obs)

        ego_obs = obs.information.get("Ego", None)
        # if ego_obs is None:
        #     raise RuntimeError(f"[❌ ERROR] Vehicle {self.vehicle.id}: 'Ego' data is missing in observation!")
        # else:
        #     print(f"[✅ OBS SET] Vehicle {self.vehicle.id}: Ego = {ego_obs}")

    def _ground_z_vehicle(self, env, x, y, default_z):
        """Return road surface Z at (x, y)."""
        try:
            m = env.world.get_map()
            loc = carla.Location(x=x, y=y, z=default_z)
            wp = m.get_waypoint(loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            return wp.transform.location.z if wp is not None else default_z
        except Exception:
            return default_z

    def _apply_physics_control(self, acc, steer_cmd):
        """Map (acc, lateral) to VehicleControl and apply when physics is enabled."""
        ctrl = carla.VehicleControl()

        # throttle / brake mapping (tune as needed)
        if acc > 0.0:
            ctrl.throttle = float(np.clip(acc / 3.0, 0.0, 1.0))
            ctrl.brake = 0.0
        else:
            ctrl.throttle = 0.0
            ctrl.brake = float(np.clip(-acc / 8.0, 0.0, 1.0))

        # steer mapping
        steer_map = {"left": -0.3, "right": 0.3, "central": 0.0}
        ctrl.steer = float(steer_map.get(steer_cmd, 0.0))

        # apply and optionally sync caches from world
        self.vehicle.apply_control(ctrl)
        try:
            self.cached_transform = self.vehicle.get_transform()
            v = self.vehicle.get_velocity()
            self.cached_velocity = carla.Vector3D(v.x, v.y, v.z)
        except Exception:
            pass

    def set_cached(self, transform=None, velocity=None):
        """Cache initial pose/velocity for later batched apply.
        Pass only what you have; None means 'leave unchanged'."""
        if transform is not None:
            self.cached_transform = transform
        if velocity is not None:
            self.cached_velocity = velocity
        return self

