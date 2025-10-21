import numpy as np
import carla
import math
from mtlsp.observation.pedestrian_observation_carla import PedestrianObservationCarla
class PedestrianWrapper:
    def __init__(self, ped, env):
        self.pedestrian = ped
        self.controller = None
        self.control_flag =False
        self.observation = None
        self.id = ped.id
        self.role = None
        self.controlled_duration = 0
        self.simulate_physics_enabled = True
        self.cached_transform = None
        self.cached_velocity = None
        self.env = env
        self.world = env.world

    def install_controller(self, controller):
        self.controller = controller
        if hasattr(self.controller, 'attach_to_pedestrian'):
            controller.attach_to_pedestrian(self)

    def reset_control_state(self):
        if self.controller and hasattr(self.controller, 'reset'):
            self.controller.reset()
        self.control_flag = False
        self.controlled_duration = 0

    def step(self):
        if self.controller and hasattr(self.controller, 'step'):
            self.controller.step()

    def is_action_legal(self,env, action):
        """Basic sanity check for WalkerControl."""
        if not isinstance(action, carla.WalkerControl):
            return False
        d = action.direction
        s = float(action.speed)
        if not np.isfinite([d.x, d.y, d.z, s]).all():
            return False
        if s < 0.0:
            return False
        return True

    def update(self, env, dt=0.05):
        """Manual integrate pose from controller.action (WalkerControl)."""
        if not self.controller or getattr(self.controller, "action", None) is None:
            return
        # action = self.controller.action  # expected: carla.WalkerControl
        #
        # # Bootstrap cache from world once
        # if self.cached_transform is None:
        #     try:
        #         self.cached_transform = self.pedestrian.get_transform()
        #     except Exception:
        #         return
        #
        # # Extract and normalize direction
        # dx, dy = float(action.direction.x), float(action.direction.y)
        # speed = float(action.speed)
        # n = (dx * dx + dy * dy) ** 0.5
        # if n <= 1e-9 or speed <= 1e-9 or dt <= 0.0:
        #     # No movement this tick; keep pose
        #     try:
        #         self.pedestrian.set_transform(self.cached_transform)
        #     except Exception:
        #         pass
        #     return
        # ux, uy = dx / n, dy / n
        #
        # # Optional caps
        # if hasattr(self, "max_speed"):
        #     speed = min(speed, float(self.max_speed))
        # step_dist = speed * dt
        # if hasattr(self, "max_step_displacement"):
        #     step_dist = min(step_dist, float(self.max_step_displacement))
        #
        # # Integrate in world frame (planar)
        # loc = self.cached_transform.location
        # new_x = loc.x + ux * step_dist
        # new_y = loc.y + uy * step_dist
        # new_z = loc.z  # keep z unchanged
        #
        # # Face along motion (optional)
        #
        # yaw_deg = math.degrees(math.atan2(uy, ux))
        # new_rot = carla.Rotation(yaw=yaw_deg, pitch=0.0, roll=0.0)
        #
        # # Update cache and push to engine
        # self.cached_transform = carla.Transform(carla.Location(new_x, new_y, new_z), new_rot)
        # try:
        #     self.pedestrian.set_transform(self.cached_transform)
        # except Exception:
        #     return
        #
        # # Lightweight bookkeeping
        # if self.cached_velocity is None:
        #     try:
        #         v = self.pedestrian.get_velocity()
        #         self.cached_velocity = carla.Vector3D(v.x, v.y, v.z)
        #     except Exception:
        #         self.cached_velocity = carla.Vector3D(0.0, 0.0, 0.0)

        self.control_flag = True
        self.controlled_duration = getattr(self, "controlled_duration", 0) + 1

    def set_role(self, role_name):
        self.role = role_name

    def set_observation(self, obs):
        self.observation = obs

    def update_observation(self, env, time_stamp=None):
        if time_stamp is None:
            snap = self.world.get_snapshot() if self.world is not None else None
            time_stamp = int(snap.frame) if snap is not None else env.get_simulation_time()
        try:
            obs = PedestrianObservationCarla(target_ped_id=self.id, time_stamp=time_stamp, traj_store=self.env.ped_traj_store)
        except TypeError:
            # fallback for older signature without env kwarg
            obs = PedestrianObservationCarla(target_ped_id=self.id, time_stamp=time_stamp,traj_store=self.env.ped_traj_store)
            # if class exposes an initializer, call it; else skip silently
            init_fn = getattr(obs, "initialize", None) or getattr(obs, "ensure_store", None)
            if callable(init_fn):
                try:
                    init_fn(env)
                except Exception:
                    pass

        # normal update
        obs.update(env)
        self.set_observation(obs)
        # obs = PedestrianObservationCarla(target_ped_id=self.id, time_stamp=time_stamp)
        # obs.update(env)
        # self.set_observation(obs)