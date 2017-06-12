import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))

from training.configs.debug_risk_env_config import Config

class TestConfig(Config):
    def __init__(self):
        super(TestConfig, self).__init__()
        self.testing = True