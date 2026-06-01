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

# Try to use the faster C++ libsumo, fallback to standard traci if it fails
try:
    import libsumo as traci
except ImportError:
    import traci

# import traci # for gui uncomment this and comment out the above import block

import matplotlib.pyplot as plt
import torch
from ctde_agent import DQNAgent

# Training hyperparameters and environment config
EPISODES = 250
STEPS_PER_EPISODE = 3600
SUMO_CMD = ["sumo", "-c", "../../env/simple.sumocfg", "--random", "--no-warnings", "--no-step-log"]
YELLOW_DURATION = 3
MIN_GREEN_DURATION = 5
MAX_GREEN_DURATION = 40

ABBR = "31_05"
DESC = "Graphical Run"

def get_dynamic_neighbors(tls_ids):
    # Find adjacent traffic lights by tracing connecting edges
    neighbors = {t: set() for t in tls_ids}
    for tls in tls_ids:
        lanes = traci.trafficlight.getControlledLanes(tls)
        for lane in lanes:
            edge = traci.lane.getEdgeID(lane)
            from_node = traci.edge.getFromJunction(edge) 
            if from_node in tls_ids and from_node != tls:
                neighbors[tls].add(from_node)
                neighbors[from_node].add(tls)
    return {k: list(v) for k, v in neighbors.items()}

def run_episode(episode_idx, agents, neighbor_map, master_agent):
    print(f"Starting episode {episode_idx + 1}/{EPISODES}")
    traci.start(SUMO_CMD)

    # Reset all agents for the new run
    for agent in agents.values():
        agent.reset()

    episode_reward = 0
    episode_teleports = 0
    episode_delay = 0
    episode_switches = 0

    last_phases = {tls_id: 0 for tls_id in agents}

    for step in range(STEPS_PER_EPISODE):
        # Step the physics engine
        traci.simulationStep()
        step_reward = 0

        episode_teleports += traci.simulation.getStartingTeleportNumber()
        
        # Collect live traffic stats before agents decide their next move
        for agent in agents.values():
            agent.collect_observations()
            episode_delay += agent.stats.get('total_queue', 0)
        
        for agent in agents.values():
            filtered_upstream = [0, 0, 0, 0]
            neighbor_phases = [0.0, 0.0, 0.0, 0.0]
            n_dirs = neighbor_map[agent.id]

            # Build the implicit 'radar' state from neighbors
            if n_dirs['N']: 
                n_agent = agents[n_dirs['N']]
                filtered_upstream[0] += n_agent.stats.get('in_n', 0)
                if n_agent.current_phase == 0: neighbor_phases[0] = 1.0
            if n_dirs['S']:
                s_agent = agents[n_dirs['S']]
                filtered_upstream[1] += s_agent.stats.get('in_s', 0)
                if s_agent.current_phase == 0: neighbor_phases[1] = 1.0
            if n_dirs['E']: 
                e_agent = agents[n_dirs['E']]
                filtered_upstream[2] += e_agent.stats.get('in_e', 0)
                if e_agent.current_phase == 4: neighbor_phases[2] = 1.0
            if n_dirs['W']: 
                w_agent = agents[n_dirs['W']]
                filtered_upstream[3] += w_agent.stats.get('in_w', 0)
                if w_agent.current_phase == 4: neighbor_phases[3] = 1.0

            strict_local_reward = agent.get_reward()

            step_reward += (strict_local_reward / 100.0)
            agent.accumulated_reward += strict_local_reward

            # Agent takes an action (or is forced to switch by the gatekeeper)
            agent.step(filtered_upstream, neighbor_phases)

            if agent.current_phase != last_phases[agent.id]:
                episode_switches += 1
            last_phases[agent.id] = agent.current_phase

        episode_reward += step_reward

        # Train the global brain using the shared replay buffer
        master_agent.learn()

    traci.close()
    return episode_reward, episode_teleports, episode_delay, episode_switches

def save_plot(data, title, line_color, ylabel, name):
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker='o', linestyle='-', color=line_color)
    plt.title(f'Learning Curve: {title} ({ABBR} - {DESC})')
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(f"../../outputs/learning_curves/{ABBR}/{name}.png")

if __name__ == "__main__":
    # Spin up SUMO briefly just to map the intersection topology
    traci.start(SUMO_CMD)
    tls_ids = traci.trafficlight.getIDList()
    neighbors = get_dynamic_neighbors(tls_ids)
    positions = {tls: traci.junction.getPosition(tls) for tls in tls_ids}

    # CTDE Setup: One master brain
    master_agent = DQNAgent(tls_id="MASTER_BRAIN")
    shared_policy = master_agent.policy_net
    shared_target = master_agent.target_net
    shared_memory = master_agent.memory

    # Local actors pointing their references to the master brain
    agents = {}
    for tls_id in tls_ids:
        agents[tls_id] = DQNAgent(tls_id, YELLOW_DURATION, MIN_GREEN_DURATION, MAX_GREEN_DURATION)
        agents[tls_id].get_static_data()

        agents[tls_id].policy_net = shared_policy
        agents[tls_id].target_net = shared_target
        agents[tls_id].memory = shared_memory

    traci.close()

    # Build a geometric map to assign N/S/E/W based on raw X/Y coordinates
    neighbor_map = {tls: {'N': None, 'S': None, 'E': None, 'W': None} for tls in tls_ids}
    
    for tls_id in tls_ids:
        ax, ay = positions[tls_id]
        for n_id in neighbors[tls_id]:
            nx, ny = positions[n_id]
            if ny > ay + 10:   neighbor_map[tls_id]['N'] = n_id
            elif ny < ay - 10: neighbor_map[tls_id]['S'] = n_id
            elif nx > ax + 10: neighbor_map[tls_id]['E'] = n_id
            elif nx < ax - 10: neighbor_map[tls_id]['W'] = n_id

    rewards_per_episode = []
    teleports_per_episode = []
    delays_per_episode = []
    switches_per_episode = []

    # Main training loop
    for i in range(EPISODES):
        episode_reward, episode_teleports, episode_delay, episode_switches = run_episode(i, agents, neighbor_map, master_agent)
        rewards_per_episode.append(episode_reward)
        teleports_per_episode.append(episode_teleports)
        delays_per_episode.append(episode_delay)
        switches_per_episode.append(episode_switches)

        # Decay exploration rate globally
        if master_agent.epsilon > master_agent.epsilon_min:
            master_agent.epsilon *= master_agent.epsilon_decay
        
        for agent in agents.values():
            agent.epsilon = master_agent.epsilon

        # Sync Double DQN target network periodically
        master_agent.sync_target_network()
        
        sample_agent_id = list(agents.keys())[0]
        print(f"  >> Ep {i+1} | Rwd: {episode_reward:.2f} | Delay: {episode_delay} | Switches: {episode_switches} | Teleports: {episode_teleports} | Epsilon: {master_agent.epsilon:.4f}")

    os.makedirs(f"../../outputs/learning_curves/{ABBR}", exist_ok=True)
    
    # Save training metrics
    save_plot(teleports_per_episode, "Teleports", "red", "Total Teleported Vehicles", "teleports")
    save_plot(rewards_per_episode, "Reward", "green", "Total Reward", "reward")
    save_plot(delays_per_episode, "Delay", "blue", "Cumulative Stopped Vehicles (Delay)", "delay")
    save_plot(switches_per_episode, "Switches", "purple", "Total Switches", "switches")

    save_dir = f"../../models/{ABBR}"
    os.makedirs(save_dir, exist_ok=True)

    # Save the final trained PyTorch model
    sample_agent = list(agents.values())[0]
    torch.save(sample_agent.policy_net.state_dict(), f"{save_dir}/dqn_global.pth")

    print("Training complete. Plots and model saved.")