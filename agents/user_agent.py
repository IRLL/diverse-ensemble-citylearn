import numpy as np
# from agents.sac_agent import SAC_Agent
from agents.rbc_agent_custom import SimpleRBCAgent


###################################################################
#####                Specify your agent here                  #####
###################################################################

class NOOPAgent:
    def __init__(self):
        self.action_space = {}

    @staticmethod
    def register_reset(*args, **kwargs):
        return np.array([0.0])

    @staticmethod
    def set_action_space(*args, **kwargs):
        pass

    # def set_action_space(self, agent_id, action_space):
    #     pass

    @staticmethod
    def compute_action(*args, **kwargs):
        return np.array([0.0])


UserAgent = SimpleRBCAgent
# UserAgent = NOOPAgent  # to make sure reward sums of no battery agent add up to 0
# UserAgent = SAC_Agent

