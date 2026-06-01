import os

# Inject SUMO binaries into the system path
if 'SUMO_HOME' in os.environ:
    sumo_bin = os.path.normpath(os.path.join(os.environ['SUMO_HOME'], 'bin'))
    if os.name == 'nt':
        try:
            os.add_dll_directory(sumo_bin)
        except Exception:
            pass
    if sumo_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = sumo_bin + os.pathsep + os.environ.get('PATH', '')

# Try to use the faster C++ libsumo, fallback to standard traci
try:
    import libsumo as traci
except ImportError:
    import traci

# import traci # for gui uncomment this and comment out the above import block

import traci.constants as tc

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()

        # Shared feature extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Calculates the baseline value of the state itself
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Calculates the specific advantage of each possible action
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Aggregation layer with mean centering to ensure identifiability
        return values + (advantages - advantages.mean(dim=1, keepdim=True))
    
class ReplayBuffer:
    # Circular buffer for storing and sampling past experiences
    def __init__(self, capacity=600000):
        self.buffer = []
        self.capacity = capacity
        self.ptr = 0

    def append(self, experience):
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.ptr] = experience
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, tls_id, yellow_duration=3, min_green_duration=5, max_green_duration=40):
        self.id = tls_id
        self.yellow_duration = yellow_duration
        self.min_green_duration = min_green_duration
        self.max_green_duration = max_green_duration
        
        # Learning hyperparameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.gamma = 0.99
        self.learning_rate = 0.00025
        self.batch_size = 256 
        self.patience_limit = 100

        self.state_dim = 29
        self.action_dim = 4

        self.target_phase = 0
        self.accumulated_reward = 0

        # Auto-detect GPU for PyTorch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Double DQN setup: Active policy net and frozen target net
        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) 
        self.target_net.eval()  
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        
        self.memory = ReplayBuffer(capacity=600000)

        # Environment state trackers
        self.current_phase = 0 
        self.time_in_phase = 0
        self.last_state = None
        self.last_action = None
        self.stats = {}

        self.step_counter = 0
        self.sync_frequency = 2000

        self.last_queue = 0
        self.starvation_trackers = {0: 0, 2: 0, 4: 0, 6: 0}

    def reset(self):
        # Wipe episode-specific trackers back to baseline
        self.current_phase = 0 
        self.target_phase = 0
        self.time_in_phase = 0
        self.last_state = None
        self.last_action = None
        self.accumulated_reward = 0
        self.stats = {}
        self.last_queue = 0
        self.starvation_trackers = {0: 0, 2: 0, 4: 0, 6: 0}

    def get_static_data(self):
        # Map physical intersection geometry to logical directions (N/S/E/W)
        self.incoming_lanes = []
        lanes = set(traci.trafficlight.getControlledLanes(self.id))
        
        for lane in lanes:
            lane_len = traci.lane.getLength(lane)
            shape = traci.lane.getShape(lane)
            dx = shape[-1][0] - shape[0][0]
            dy = shape[-1][1] - shape[0][1]
            is_vertical = abs(dy) > abs(dx)
            is_left = lane.endswith("_1")
            
            if is_vertical: direction = 'in_n' if dy < 0 else 'in_s'
            else:           direction = 'in_w' if dx > 0 else 'in_e'
                
            self.incoming_lanes.append({
                'id': lane, 'len': lane_len, 
                'is_vertical': is_vertical, 'is_left': is_left, 
                'direction': direction
            })
            
        self.outgoing_lanes = []
        links = traci.trafficlight.getControlledLinks(self.id)
        out_lane_ids = set([link_group[0][1] for link_group in links if link_group])
                
        for lane in out_lane_ids:
            shape = traci.lane.getShape(lane)
            is_vertical = abs(shape[-1][1] - shape[0][1]) > abs(shape[-1][0] - shape[0][0])
            self.outgoing_lanes.append({'id': lane, 'is_vertical': is_vertical})

    def collect_observations(self):
        # Scrape live traffic data from SUMO loop detectors
        self.current_phase = traci.trafficlight.getPhase(self.id)
        data = {'q_ns_s': 0, 'q_ns_l': 0, 'q_ew_s': 0, 'q_ew_l': 0,
                'a_ns': 0, 'a_ew': 0, 'total_queue': 0,
                'down_ns': 0, 'down_ew': 0,
                'near_ns_s': 0, 'near_ns_l': 0, 'near_ew_s': 0, 'near_ew_l': 0, 'in_n': 0, 'in_s': 0, 'in_e': 0, 'in_w': 0}

        for lane_data in self.incoming_lanes:
            lane = lane_data['id']
            
            detector_id = f"e2_{lane}"
            near_vehs = traci.lanearea.getLastStepVehicleNumber(detector_id)

            res = traci.lane.getSubscriptionResults(lane)
            
            # Subscribe to SUMO variables if not already tracking them
            if not res:
                traci.lane.subscribe(lane, [
                    tc.LAST_STEP_VEHICLE_HALTING_NUMBER,
                    tc.LAST_STEP_VEHICLE_NUMBER
                ])
                halting_num = traci.lane.getLastStepHaltingNumber(lane)
                total_vehs = traci.lane.getLastStepVehicleNumber(lane)
            else:
                halting_num = res.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
                total_vehs = res.get(tc.LAST_STEP_VEHICLE_NUMBER, 0)

            data['total_queue'] += halting_num

            if lane_data['is_vertical']:
                if lane_data['is_left']:
                    data['q_ns_l'] += halting_num
                    data['near_ns_l'] += near_vehs
                else:
                    data['q_ns_s'] += halting_num
                    data['near_ns_s'] += near_vehs
            else:
                if lane_data['is_left']:
                    data['q_ew_l'] += halting_num
                    data['near_ew_l'] += near_vehs
                else:
                    data['q_ew_s'] += halting_num
                    data['near_ew_s'] += near_vehs

            moving_vehs = total_vehs - halting_num

            if lane_data['is_vertical']:    data['a_ns'] += moving_vehs
            else:                           data['a_ew'] += moving_vehs

            data[lane_data['direction']] += moving_vehs

        for lane_data in self.outgoing_lanes:
            lane = lane_data['id']
            res = traci.lane.getSubscriptionResults(lane)
            
            if not res:
                traci.lane.subscribe(lane, [tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
                halting_num = traci.lane.getLastStepHaltingNumber(lane)
            else:
                halting_num = res.get(tc.LAST_STEP_VEHICLE_HALTING_NUMBER, 0)
                
            if lane_data['is_vertical']: data['down_ns'] += halting_num
            else:                        data['down_ew'] += halting_num
        
        self.stats = data

        # Update starvation trackers for lanes currently experiencing a red light
        queue_mapping = {
            0: data['q_ns_s'], 
            2: data['q_ns_l'], 
            4: data['q_ew_s'], 
            6: data['q_ew_l']
        }
        
        for phase, queue_size in queue_mapping.items():
            if queue_size > 0 and self.current_phase != phase:
                self.starvation_trackers[phase] += 1
            else:
                self.starvation_trackers[phase] = 0

    def get_state(self, filtered_upstream, neighbor_phases):
        CRITICAL_CAPACITY = 100
        TELEPORT_LIMIT = 300

        # One-hot encode the current traffic light phase (4 possible green phases)
        p_0 = 1.0 if self.current_phase == 0 else 0.0
        p_2 = 1.0 if self.current_phase == 2 else 0.0
        p_4 = 1.0 if self.current_phase == 4 else 0.0
        p_6 = 1.0 if self.current_phase == 6 else 0.0

        # Normalize raw traffic data into a strict [0, 1] range for the neural network
        norm_time = min(self.time_in_phase / 40.0, 1.0)
        
        norm_q_ns_s = min(self.stats['q_ns_s'] / CRITICAL_CAPACITY, 1.0)
        norm_q_ns_l = min(self.stats['q_ns_l'] / CRITICAL_CAPACITY, 1.0)
        norm_q_ew_s = min(self.stats['q_ew_s'] / CRITICAL_CAPACITY, 1.0)
        norm_q_ew_l = min(self.stats['q_ew_l'] / CRITICAL_CAPACITY, 1.0)

        norm_a_ns = min(self.stats['a_ns'] / 30.0, 1.0)
        norm_a_ew = min(self.stats['a_ew'] / 30.0, 1.0)

        norm_down_ns = min(self.stats['down_ns'] / CRITICAL_CAPACITY, 1.0)
        norm_down_ew = min(self.stats['down_ew'] / CRITICAL_CAPACITY, 1.0)

        norm_up_n = min(filtered_upstream[0] / CRITICAL_CAPACITY, 1.0)
        norm_up_s = min(filtered_upstream[1] / CRITICAL_CAPACITY, 1.0)
        norm_up_e = min(filtered_upstream[2] / CRITICAL_CAPACITY, 1.0)
        norm_up_w = min(filtered_upstream[3] / CRITICAL_CAPACITY, 1.0)

        norm_starve_0 = min(self.starvation_trackers[0] / TELEPORT_LIMIT, 1.0)
        norm_starve_2 = min(self.starvation_trackers[2] / TELEPORT_LIMIT, 1.0)
        norm_starve_4 = min(self.starvation_trackers[4] / TELEPORT_LIMIT, 1.0)
        norm_starve_6 = min(self.starvation_trackers[6] / TELEPORT_LIMIT, 1.0)

        norm_near_ns_s = min(self.stats['near_ns_s'] / 10.0, 1.0)
        norm_near_ns_l = min(self.stats['near_ns_l'] / 10.0, 1.0)
        norm_near_ew_s = min(self.stats['near_ew_s'] / 10.0, 1.0)
        norm_near_ew_l = min(self.stats['near_ew_l'] / 10.0, 1.0)

        state_array = np.array([
            p_0, p_2, p_4, p_6, norm_time,
            norm_q_ns_s, norm_q_ns_l, norm_q_ew_s,
            norm_q_ew_l, norm_a_ns, norm_a_ew,
            norm_down_ns, norm_down_ew,
            norm_up_n, norm_up_s, norm_up_e, norm_up_w,
            neighbor_phases[0], neighbor_phases[1],
            neighbor_phases[2], neighbor_phases[3],
            norm_starve_0, norm_starve_2, norm_starve_4, norm_starve_6,
            norm_near_ns_s, norm_near_ns_l, norm_near_ew_s, norm_near_ew_l
        ], dtype=np.float32)

        return state_array

    def get_reward(self):
        # 1. Calculate System Penalty
        total_queue = self.stats.get('total_queue', 0)
        system_penalty = total_queue / 40.0

        # 2. Calculate Quadratic Starvation Penalty
        starvation_penalty = 0.0
        phase_to_queue = {
            0: 'q_ns_s',
            2: 'q_ns_l',
            4: 'q_ew_s',
            6: 'q_ew_l'
        }
        for phase, wait_time in self.starvation_trackers.items():
            if wait_time > self.patience_limit:
                overtime = wait_time - self.patience_limit
                overtime_factor = (overtime / 25.0) ** 2

                queue_key = phase_to_queue[phase]
                phase_queue_length = self.stats.get(queue_key, 0)
                queue_factor = min(max(phase_queue_length / 10.0, 0.1), 2.0)

                starvation_penalty += overtime_factor * queue_factor

        # 3. Calculate Wasted Green Penalty
        wasted_green_penalty = 0.0
        competing_queue = 0
        if self.current_phase == 0 and self.stats.get('near_ns_s', 0) == 0:
            competing_queue = self.stats.get('q_ns_l', 0) + self.stats.get('q_ew_s', 0) + self.stats.get('q_ew_l', 0)
        elif self.current_phase == 2 and self.stats.get('near_ns_l', 0) == 0:
            competing_queue = self.stats.get('q_ns_s', 0) + self.stats.get('q_ew_s', 0) + self.stats.get('q_ew_l', 0)
        elif self.current_phase == 4 and self.stats.get('near_ew_s', 0) == 0:
            competing_queue = self.stats.get('q_ns_s', 0) + self.stats.get('q_ns_l', 0) + self.stats.get('q_ew_l', 0)
        elif self.current_phase == 6 and self.stats.get('near_ew_l', 0) == 0:
            competing_queue = self.stats.get('q_ns_s', 0) + self.stats.get('q_ns_l', 0) + self.stats.get('q_ew_s', 0)

        if competing_queue > 0:
            wasted_green_penalty = (competing_queue / 40.0) * 0.5

        return - (system_penalty + starvation_penalty + wasted_green_penalty)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def choose_action(self, state):
        # Epsilon-greedy: Explore randomly or exploit the neural network's prediction
        if random.random() < self.epsilon:
            return random.choice([0, 1, 2, 3])
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        
        return torch.argmax(q_values).item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Random mini-batch sampling to stabilize learning
        batch = self.memory.sample(self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states = zip(*batch)

        states = torch.FloatTensor(np.vstack(batch_states)).to(self.device)
        actions = torch.LongTensor(batch_actions).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(batch_rewards).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.vstack(batch_next_states)).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Double DQN logic: Decoupled action selection and evaluation
        with torch.no_grad():
            best_next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            max_next_q_values = self.target_net(next_states).gather(1, best_next_actions)
            target_q_values = rewards + (self.gamma * max_next_q_values)
            
        # Huber Loss with Gradient Clipping to prevent exploding gradients
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

    def sync_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def step(self, filtered_upstream, neighbor_phases, eval_mode=False):
        # 1. Enforce mandatory yellow transition phases
        if self.current_phase % 2 != 0: 
            if self.time_in_phase >= self.yellow_duration:
                traci.trafficlight.setPhase(self.id, self.target_phase)
                self.current_phase = self.target_phase
                self.time_in_phase = 0
            else:
                self.time_in_phase += 1
            return
            
        # 2. Enforce minimum green duration
        if self.min_green_duration >  self.time_in_phase >= 0:
            self.time_in_phase += 1
            return
        
        # 3. Collect observations
        current_state = self.get_state(filtered_upstream, neighbor_phases)

        # 4. Choose action based on max green duration or NN query
        if self.time_in_phase >= self.max_green_duration:
            requested_phase = (self.current_phase + 2) % 8
            action = requested_phase // 2
        else:
            action = self.choose_action(current_state)
            requested_phase = action * 2

        # 5. Handle memory
        if not eval_mode:
            scaled_reward = self.accumulated_reward / 100.0
            if self.last_state is not None:
                self.remember(self.last_state, self.last_action, scaled_reward, current_state)

        # 6. Execute phase change logic
        if requested_phase != self.current_phase:
            self.target_phase = requested_phase
            yellow_phase = self.current_phase + 1
            traci.trafficlight.setPhase(self.id, yellow_phase)
            
            self.current_phase = yellow_phase
            self.time_in_phase = 0
        else:
            self.time_in_phase += 1
        
        # 7. Update trackers
        self.accumulated_reward = 0
        self.last_state = current_state
        self.last_action = action
        self.step_counter += 1