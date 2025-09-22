from dataclasses import dataclass
from collections import defaultdict, deque
import math


# Default Configuration for Trajectory Storage
@dataclass
class Obs_Config:
    hist_frames: int = 100          # frames to keep the data per pedestrian
    neighbor_radius: float = 50.0   # radius for the ego-ped defines its neighbors (meters)
    topk_neighbors: int = 20        # top k neighbors within the radius for the ego-ped



class RingBuffer:
    '''
    Ring Buffer: 
        Store past data for a single pedestrian
        Data structure: Double-ended queue (deque) -> One side for appending new data, the other side for popping old data
        Data type for each entry: (timestamp, x, y) -> dimension = 3 for each row
    '''
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        self.DIM = 3
    # Add new row of data to the end of the buffer
    def push(self, row):
        self.buffer.append(row)
    
    # Remove the oldest row of data from the front of the buffer
    def popleft(self):
        self.buffer.popleft()
    


class TrajStore:
    '''
    Trajectory Storage: 
        A buffer for every pedestrian model
        Data structure: Dictionary of RingBuffer, key is pedestrian ID
    '''
    def __init__(self, config: Obs_Config):
        self.config = config
        self.ringbuffer = {}        # {pid: RingBuffer}
        self.latest_positions = {}  # {pid: (x, y)}
        self._last_frame = -1

    def update(self, pid, frame: int, x: float, y: float):
        # Initialize ring buffer for new pedestrian
        if pid not in self.ringbuffer:
            self.ringbuffer[pid] = RingBuffer(capacity=self.config.hist_frames)

        # Add new data point
        self.ringbuffer[pid].push((frame, x, y))
        self.latest_positions[pid] = (x, y)

        # Remove old data points beyond hist_frames
        while self.ringbuffer[pid].buffer and ((frame - self.ringbuffer[pid].buffer[0][0]) > self.config.hist_frames):
            self.ringbuffer[pid].popleft()
    
    # Get the neighbors for the ego pedestrian
    def get_neighbors(self, ego_id, radius, topk):
        if radius is None:
            radius = self.config.neighbor_radius
        if topk is None:
            topk = self.config.topk_neighbors
        if ego_id not in self.latest_positions:
            return []
        
        # Get ego pedestrian's latest position
        ego_x, ego_y = self.latest_positions[ego_id]
        neighbors = []

        # Compute distances to surrounding pedestrians, append those within the radius
        for (pid, position) in self.latest_positions.items():
            if (pid == ego_id):
                continue
            x, y = position
            distance = math.sqrt((x - ego_x)**2 + (y - ego_y)**2)
            if distance <= radius:
                neighbors.append((pid, distance))
            
        neighbors.sort(key=lambda a: a[1])  # Sort by distance
        topK_neighbors_id = [pid for pid, _ in neighbors[:topk]]   # Get top-k closest neighbors
        return topK_neighbors_id
    
    # Scan the environment if needed (implemented in new frame)
    def scan_if_needed(self, env=None):
        if not env:
            raise ValueError("No environment is provided!")
        elif not env.world:
            raise ValueError("No world is provided!")
        
        snapshot = env.world.get_snapshot()
        frame = snapshot.frame

        if (frame == self._last_frame):
            return False
        pedestrian_list = env.world.get_actors().filter('walker.pedestrian.*')
        for pedestrian in pedestrian_list:
            transform = pedestrian.get_transform()
            self.update(pedestrian.id, frame, transform.location.x, transform.location.y)
        
        self._last_frame = frame
        return True

    
    # Build observation for all pedestrians
    def build_observation(self, ego_id, relative=False):

        if ego_id not in self.ringbuffer or not self.ringbuffer[ego_id].buffer:
            return {
                "Ego-ped ID": ego_id,
                "Ego Past Trajectory": [],
                "Neighbors": [],
                "Neighbors' Past Trajectories": {}
            }
        
        # get the ego pedestrian's trajectory by frames
        ego_traj = list(self.ringbuffer[ego_id].buffer)
        ego_traj_dict = {}
        for frame, ego_x, ego_y in ego_traj:
            ego_traj_dict[frame] = (ego_x, ego_y)

        neighbors_id = self.get_neighbors(ego_id, self.config.neighbor_radius, self.config.topk_neighbors)
        neighbors_past_traj = defaultdict(list)
        
        # use different neighbors' past trajectory (relative position or global position)
        if relative:
            for nid in neighbors_id:
                if nid in self.ringbuffer:
                    neighbor_traj = list(self.ringbuffer[nid].buffer)
                    neighbor_traj_dict = {}
                    for frame, nx, ny in neighbor_traj:
                        neighbor_traj_dict[frame] = (nx, ny)
                    
                    relative_traj_list = []
                    # Convert to relative position
                    for frame in ego_traj_dict.keys():
                        if frame in neighbor_traj_dict:
                            ex, ey = ego_traj_dict[frame]
                            nx, ny = neighbor_traj_dict[frame]
                            relative_traj_list.append((frame, nx - ex, ny - ey))

                    neighbors_past_traj[nid] = relative_traj_list

                else:
                    neighbors_past_traj[nid] = []
        
        else:
            # Global coordinates
            for nid in neighbors_id:
                if nid in self.ringbuffer:
                    neighbors_past_traj[nid] = list(self.ringbuffer[nid].buffer)
                else:
                    neighbors_past_traj[nid] = []

        neighbors_past_traj = dict(neighbors_past_traj)

        return {"Ego-ped ID": ego_id,
                "Ego Past Trajectory": ego_traj,
                "Neighbors": neighbors_id,
                "Neighbors' Past Trajectories": neighbors_past_traj,
                }
