import unittest
from unittest.mock import patch

from ctde_agent import ReplayBuffer, DQNAgent 

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        # Initialize a tiny buffer for easy testing
        self.buffer = ReplayBuffer(capacity=3)

    def test_capacity_and_circular_overwrite(self):
        self.buffer.append(1)
        self.buffer.append(2)
        self.buffer.append(3)
        self.assertEqual(len(self.buffer), 3)
        
        # This 4th item should overwrite the 1st item at index 0
        self.buffer.append(4)
        self.assertEqual(len(self.buffer), 3)
        self.assertEqual(self.buffer.buffer[0], 4)
        self.assertEqual(self.buffer.buffer[1], 2)

    def test_sample_batch(self):
        for i in range(10):
            self.buffer.append(i)
        
        # Ensure sampling returns the exact batch size requested
        sample = self.buffer.sample(batch_size=2)
        self.assertEqual(len(sample), 2)

class TestAgentMath(unittest.TestCase):
    def setUp(self):
        # Instantiate the agent. 
        # (This is safe because __init__ does not call SUMO directly)
        self.agent = DQNAgent(tls_id="test_node")
        
        # Inject a baseline, empty traffic state
        self.agent.stats = {
            'q_ns_s': 0, 'q_ns_l': 0, 'q_ew_s': 0, 'q_ew_l': 0,
            'total_queue': 0, 'near_ns_s': 0, 'near_ns_l': 0,
            'near_ew_s': 0, 'near_ew_l': 0, 'down_ns': 0, 'down_ew': 0,
            'a_ns': 0, 'a_ew': 0
        }
        self.agent.starvation_trackers = {0: 0, 2: 0, 4: 0, 6: 0}

    def test_get_state_clamping_and_one_hot(self):
        self.agent.current_phase = 2
        
        # Inject a massive queue (500 cars) that exceeds critical capacity
        self.agent.stats['q_ns_s'] = 500 
        
        state = self.agent.get_state(filtered_upstream=[0,0,0,0], neighbor_phases=[0,0,0,0])
        
        # 1. Assert Tensor Shape is exactly 29
        self.assertEqual(state.shape, (29,))
        
        # 2. Assert One-Hot Encoding (Phase 2 should be 1.0, Phase 0 should be 0.0)
        self.assertEqual(state[0], 0.0) # p_0
        self.assertEqual(state[1], 1.0) # p_2
        
        # 3. Assert Strict Clamping (norm_q_ns_s is at index 5)
        # Even with 500 cars, the neural network should only see 1.0 max
        self.assertEqual(state[5], 1.0)

    def test_get_reward_starvation_penalty(self):
        # Inject exactly 25 seconds of overtime on phase 0
        self.agent.starvation_trackers[0] = 125 
        
        # Inject 20 cars waiting (which creates a queue factor of 2.0)
        self.agent.stats['q_ns_s'] = 20 
        self.agent.stats['total_queue'] = 20
        
        reward = self.agent.get_reward()
        
        # overtime_factor = (25 / 25.0)^2 = 1.0
        # queue_factor = min(max(20 / 10.0, 0.1), 2.0) = 2.0
        # starvation_penalty = 1.0 * 2.0 = 2.0
        # system_penalty = 20 / 40.0 = 0.5
        # Total Reward = -(2.0 + 0.5) = -2.5
        
        self.assertEqual(reward, -2.5)

class TestSafetyGatekeeper(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAgent(tls_id="test_node")
        self.agent.stats = {
            'q_ns_s': 0, 'q_ns_l': 0, 'q_ew_s': 0, 'q_ew_l': 0,
            'total_queue': 0, 'near_ns_s': 0, 'near_ns_l': 0,
            'near_ew_s': 0, 'near_ew_l': 0, 'down_ns': 0, 'down_ew': 0,
            'a_ns': 0, 'a_ew': 0
        }
        self.agent.starvation_trackers = {0: 0, 2: 0, 4: 0, 6: 0}

    @patch('ctde_agent.traci') 
    def test_gatekeeper_enforces_min_green(self, mock_traci):
        self.agent.current_phase = 0
        self.agent.time_in_phase = 2  # Has not hit the min_green of 5
        self.agent.min_green_duration = 5

        self.agent.step(filtered_upstream=[0,0,0,0], neighbor_phases=[0,0,0,0])

        # Assert time increased, but Traci was never called to change the phase
        self.assertEqual(self.agent.time_in_phase, 3)
        mock_traci.trafficlight.setPhase.assert_not_called()

    @patch('ctde_agent.traci')
    def test_gatekeeper_forces_max_green_shift(self, mock_traci):
        self.agent.current_phase = 0
        self.agent.time_in_phase = 40 # Hits the exact max_green limit
        self.agent.max_green_duration = 40

        self.agent.step(filtered_upstream=[0,0,0,0], neighbor_phases=[0,0,0,0])

        # Assert the agent bypassed the NN, targeted phase 2, and triggered yellow phase 1
        self.assertEqual(self.agent.target_phase, 2)
        self.assertEqual(self.agent.current_phase, 1)
        self.assertEqual(self.agent.time_in_phase, 0)
        
        # Assert the exact command sent to the SUMO middleware was the yellow light
        mock_traci.trafficlight.setPhase.assert_called_with("test_node", 1)

if __name__ == '__main__':
    unittest.main()