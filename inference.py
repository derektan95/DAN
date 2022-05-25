import torch
import numpy as np
from config import config
from model_inference import Model

cfg = config()


if __name__ == '__main__':

    # Initialize Depot & Target cities
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    depot_position = np.zeros((1,1,2)) #np.random.rand(1, 1, 2)
    target_position = np.random.rand(1, cfg.target_size - 1, 2)
    target_inputs = [torch.FloatTensor(depot_position).cuda(), torch.FloatTensor(target_position).cuda()]

    # Initialize ALL agent positions
    agent_position = torch.Tensor(depot_position).to(cfg.device) # OWN positions
    all_agent_positions = torch.zeros((1, cfg.agent_amount, 2)).to(cfg.device)
    all_agent_action_gaps = torch.zeros((1, cfg.agent_amount,1)).to(cfg.device)
    all_agent_inputs = torch.cat((all_agent_positions, all_agent_action_gaps), -1)
    # all_agent_inputs = torch.Tensor([[[0.1, 0.1, 0], [0, 0, 0], [0.45, 0.0, 0]]]).cuda()   # SIMULATED AGENT POSITIONS
    
    # Initialize city visitation mask [ visited = 1, unvisited = 0 ]
    global_mask = torch.zeros((1, cfg.target_size), device=cfg.device, dtype=torch.int64)
    # global_mask[0][0] = 1      # FORCE FIRST CITY TO BE VISITED, COS (0,0) IS WHERE ROBOTS INIT AT

    # Loads model & weights
    checkpoint = torch.load(cfg.model_path + '/model_states.pth')
    model = Model(cfg, decode_type=cfg.strategy, training=False)
    model.to(cfg.device)
    model.load_state_dict(checkpoint['model'])

    # Forward Pass
    with torch.no_grad():    
        next_target_idx = model(target_inputs, agent_position, all_agent_inputs, global_mask)
    
    print("Next Target (Index): ", next_target_idx.item())







