import unittest
import torch
import numpy as np
from src.high_level.high_controller import HighLevelController
from src.mid_level.mid_controller import MidLevelController
from src.low_level.low_controller import LowLevelController
from src.utils.config_loader import ConfigLoader

class TestControllers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ConfigLoader()
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def test_high_level_controller(self):
        controller = HighLevelController(self.config.config, self.device)
        
        # 测试动作选择
        state = np.random.randn(self.config.config['environment']['state_dim']['high'])
        vehicles = [{'id': 'v1', 'x': 0, 'y': 0, 'speed': 10, 'direction': 0}]
        
        action = controller.select_action(state, vehicles)
        self.assertIsInstance(action, dict)
        self.assertIn('collision_pairs', action)
        self.assertIn('allocations', action)
        self.assertIn('action_idx', action)
        
    def test_mid_level_controller(self):
        controller = MidLevelController(self.config.config, self.device)
        
        # 测试动作选择
        state = np.random.randn(self.config.config['environment']['state_dim']['mid'])
        vehicle = {'id': 'v1', 'x': 0, 'y': 0, 'speed': 10}
        collision_pairs = []
        time_allocation = {}
        nearby_vehicles = []
        
        action = controller.select_action(state, vehicle, collision_pairs, 
                                        time_allocation, nearby_vehicles)
        self.assertIsInstance(action, np.ndarray)
        self.assertEqual(action.shape[0], self.config.config['environment']['action_dim']['mid'])
        
    def test_low_level_controller(self):
        controller = LowLevelController(self.config.config, self.device)
        
        # 测试动作选择
        state = np.random.randn(self.config.config['environment']['state_dim']['low'])
        vehicle = {'id': 'v1', 'x': 0, 'y': 0, 'speed': 10}
        
        action = controller.select_action(state, vehicle)
        self.assertIsInstance(action, dict)
        self.assertIn('target_speed', action)
        self.assertIn('acceleration', action)

if __name__ == '__main__':
    unittest.main()
