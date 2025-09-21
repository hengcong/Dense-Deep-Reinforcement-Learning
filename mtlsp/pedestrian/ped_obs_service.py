from dataclasses import dataclass
from collections import defaultdict, deque
import numpy as np
import math


# Default Configuration for Trajectory Storage
@dataclass
class Obs_Config:
    hist_secs: float = 5.0          # seconds to keep the data per pedestrian
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
    
    # Turn the buffer into a stackked numpy array
    def as_stack(self) -> np.ndarray:
        if not self.buffer:
            return np.zeros((0, self.DIM), dtype=np.float32)
        return np.stack(self.buffer, axis=0, dtype=np.float32)


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
    
    def update(self, pid, t, x: float, y: float):
        # Initialize ring buffer for new pedestrian
        if pid not in self.ringbuffer:
            self.ringbuffer[pid] = RingBuffer(capacity=self.config.hist_secs * 10)  # assuming 10 Hz update rate
        
        # Add new data point
        self.ringbuffer[pid].push((t, x, y))
        self.latest_positions[pid] = (x, y)

        # Remove old data points beyond hist_secs
        while self.ringbuffer[pid].buffer and (t - self.ringbuffer[pid].buffer[0][0] > self.config.hist_secs):
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
        for pid, (x, y) in self.latest_positions:
            if (pid == ego_id):
                continue
            distance = math.sqrt((x - ego_x)**2 + (y - ego_y)**2)
            if distance <= radius:
                neighbors.append((pid, distance))
            
        neighbors.sort(key=lambda a: a[1])  # Sort by distance
        topK_neighbors = [pid for pid, _ in neighbors[:topk]]   # Get top-k closest neighbors
        return topK_neighbors


class Ped_Obs_Service:
    '''
    Pedestrian Observation Service: 
        Manage the observation for all pedestrians -> NEED TO BE DONE
        This is called before implementing pedestrian controller, 
        which means it should be activated in the environment or higher layer.
        Data structure:
    '''
    def __init__(self, traj_store: TrajStore):
        self.config = Obs_Config()
        self.traj_store = traj_store
    
    # Build observation for all pedestrians
    def build_observation(self):
        return 

    def scan_if_needed(self):
        pass
    
