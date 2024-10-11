import carla
from gym import spaces, core

class CarlaEnv(core.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()
        self.action_space = spaces.Box(low=0.001, high=0.999, shape=(1,))
        self.observation_space = spaces.Box(low=-5, high=5, shape=(10,))

        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0  # some customized metric logging

        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)  # Set the connection timeout
        self.world = self.client.load_world('Town01')  # Load a specific town
        self.blueprint_library = self.world.get_blueprint_library()

        # Spawn points and sensors can be set up here
        self.ego_vehicle = None
        self.setup_vehicle()
        self.setup_sensors()

    def setup_vehicle(self):
        # Define and spawn the ego vehicle
        vehicle_bp = self.blueprint_library.filter('vehicle.*model3*')[0]
        spawn_point = self.world.get_map().get_spawn_points()[0]
        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

    def setup_sensors(self):
        # Define sensors like cameras or LIDARs
        # Attach sensors to the vehicle and set up data listeners
        pass

    def step(self, action):
        pass

    def reset(self):
        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0

        # Destroy the previous vehicle and spawn a new one
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()

        self.setup_vehicle()
        self.setup_sensors()

        # Return the initial state
        return self.get_state()

    def get_state(self):
        pass

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
        # Check if vehicle has collided
        if self.check_collision():
            return True

        # Check if vehicle is off the road (optional)
        if self.is_off_road():
            return True

        return False
