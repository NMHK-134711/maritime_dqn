import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def train_dqn(env, agent, max_episodes=1000):
    rewards = []
    path_lengths = []
    epsilon = agent.epsilon_start
    image_dir = r"C:/baramproject/trained_model/maritime_dqn_4/episode_debug_image"
    data_dir = r"C:/baramproject/trained_model/maritime_dqn_4/episode_debug_data"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    for episode in tqdm(range(max_episodes), desc="Training Episodes"):
        state = env.reset()
        total_reward = 0
        path_length = 0
        done = False
        debug_mode = episode % 500 == 0
        debug_data = [] if debug_mode else None
        
        while not done:
            epsilon = max(agent.epsilon_end, epsilon - (agent.epsilon_start - agent.epsilon_end) / agent.epsilon_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            q_values = agent.policy_net(torch.FloatTensor(state).unsqueeze(0).to(agent.device)).detach().cpu().numpy().flatten()
            if debug_mode:
                debug_data.append({
                    "step": path_length,
                    "state": state.tolist(),
                    "action": action,
                    "reward": reward,
                    "next_state": next_state.tolist(),
                    "q_values": q_values.tolist(),
                    "epsilon": epsilon
                })
            agent.store_experience(state, action, reward, next_state, done)
            if path_length % 4 == 0:
                agent.update()
            state = next_state
            total_reward += reward
            path_length += 1
        
        rewards.append(total_reward)
        path_lengths.append(path_length)

        if debug_mode:
            with open(os.path.join(data_dir, f"episode_{episode}.json"), 'w') as f:
                json.dump(debug_data, f, indent=4)
        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}, Path Length: {path_length}")
            plt.figure(figsize=(10, 8))
            plt.imshow(env.grid, cmap='gray')
            path_array = np.array(env.path)
            plt.plot(path_array[:, 1], path_array[:, 0], 'r-', label='Path')
            plt.plot(env.start_pos[1], env.start_pos[0], 'go', label='Start')
            plt.plot(env.end_pos[1], env.end_pos[0], 'bo', label='End')
            plt.legend()
            plt.title(f"Episode {episode} Path")
            plt.savefig(os.path.join(image_dir, f"episode_{episode}.png"))
            plt.close()
   
    model_path = r"C:/baramproject/trained_model/maritime_dqn_4/navigation_model.pth"
    torch.save(agent.policy_net.state_dict(), model_path)
    return rewards, path_lengths
