import torch
import torch.nn as nn
import copy
from agentEncoder import AgentEncoder
from targetEncoder import TargetEncoder
from decoder import Decoder


class Model(nn.Module):
    def __init__(self, cfg, decode_type='sampling', training=True):
        super(Model, self).__init__()
        self.local_agent_encoder = AgentEncoder(cfg)
        self.local_target_encoder = TargetEncoder(cfg)
        self.local_decoder = Decoder(cfg)
        self.training = training
        self.decode_type = decode_type

    def forward(self, target_inputs, agent_position, agent_inputs, global_mask):

        agent_feature, _ = self.calculate_encoded_agent(agent_position, agent_inputs)
        target_feature, _, _ = self.calculate_encoded_target(agent_position, target_inputs)
        next_target_index, _ = self.local_decoder(target_feature=target_feature,
                                                    current_state=torch.mean(target_feature,dim=1).unsqueeze(1),
                                                    agent_feature=agent_feature,
                                                    mask=global_mask,
                                                    decode_type=self.decode_type,
                                                    next_target=None)        
        return next_target_index


    def calculate_encoded_agent(self, agent_position, agent_inputs):
        # print("Agent shape: ",agent_inputs.size())
        agent_inputs = agent_inputs - torch.cat((agent_position,torch.FloatTensor([[[0]]]).cuda()),dim=-1)
        # print("~~~~RELATIVE AGENT POSE: ", agent_inputs)
        agent_feature = self.local_agent_encoder(agent_inputs)
        return agent_feature, agent_inputs

    def calculate_encoded_target(self, agent_position, target_inputs_in):
        target_inputs = copy.deepcopy(target_inputs_in)
        depot_inputs = target_inputs[0] - agent_position
        city_inputs = target_inputs[1] - agent_position
        target_feature = self.local_target_encoder(depot_inputs,city_inputs)
        return target_feature,depot_inputs,city_inputs



    def get_log_p(self, _log_p, pi):
        """	args:
            _log_p: (batch, city_t, city_t)
            pi: (batch, city_t), predicted tour
            return: (batch) sum of the log probability of the chosen targets
        """
        log_p = torch.sum(torch.gather(input=_log_p, dim=2, index=pi[:, 1:, None]), dim=1)
        return log_p
