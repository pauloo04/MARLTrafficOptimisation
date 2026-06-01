import os
import datetime

if 'SUMO_HOME' in os.environ:
    sumo_bin = os.path.normpath(os.path.join(os.environ['SUMO_HOME'], 'bin'))
    if os.name == 'nt':
        try:
            os.add_dll_directory(sumo_bin)
        except Exception:
            pass
    if sumo_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = sumo_bin + os.pathsep + os.environ.get('PATH', '')

try:
    import libsumo as traci # type: ignore
    print("--- RUNNING WITH FAST NATIVE LIBSUMO ---")
except ImportError:
    import traci
    print("--- WARNING: FALLING BACK TO SLOW TRACI ---")

import traci.constants as tc
import torch # type: ignore
from ctde_agent import DQNAgent

# Evaluation Configurations
EPISODES = 10
STEPS_PER_EPISODE = 3600
SEEDS = [42, 100, 999, 1234, 5555, 11, 888, 2024, 2025, 2026]

# RL Agent Specifics
YELLOW_DURATION = 3
MIN_GREEN_DURATION = 5
MAX_GREEN_DURATION = 40
ABBR = "29_05_3"
LOAD_DIR = f'../models/{ABBR}'

# Model Configurations
MODELS = {
    "Fixed":    {"cfg": "../env/simple_fixed.sumocfg",    "is_rl": False},
    "Actuated": {"cfg": "../env/simple_actuated.sumocfg", "is_rl": False},
    "RL_Agent": {"cfg": "../env/simple.sumocfg",          "is_rl": True}
}

def get_dynamic_neighbors(tls_ids):
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

def run_episode(model_name, current_seed, agents=None, neighbor_map=None):
    cfg_path = MODELS[model_name]["cfg"]
    cmd = ["sumo", "-c", cfg_path, "--seed", str(current_seed), "--no-warnings", "--no-step-log"]
    
    traci.start(cmd)
    
    # Reset RL agents if applicable
    if agents:
        for agent in agents.values():
            agent.reset()

    # Setup Universal Metric Subscriptions
    tls_ids = traci.trafficlight.getIDList()
    controlled_lanes = set()
    for tls in tls_ids:
        for lane in traci.trafficlight.getControlledLanes(tls):
            controlled_lanes.add(lane)
            
    all_edges = traci.edge.getIDList()

    # Tracking metrics
    episode_delay = 0
    episode_teleports = 0
    episode_co2_mg = 0.0
    episode_fuel_mg = 0.0

    # Simulation Loop
    for _ in range(STEPS_PER_EPISODE):
        traci.simulationStep()
        
        episode_teleports += traci.simulation.getStartingTeleportNumber()
        
        # 1. Collect Universal Subscriptions
        for lane in controlled_lanes:
            episode_delay += traci.lane.getLastStepHaltingNumber(lane)
                
        for edge in all_edges:
            episode_co2_mg += traci.edge.getCO2Emission(edge)
            episode_fuel_mg += traci.edge.getFuelConsumption(edge)

        # 2. Execute RL Actions (If Applicable)
        if agents and neighbor_map:
            for agent in agents.values():
                agent.collect_observations()
            
            for agent in agents.values():
                filtered_upstream = [0, 0, 0, 0]
                neighbor_phases = [0.0, 0.0, 0.0, 0.0]
                n_dirs = neighbor_map[agent.id]

                if n_dirs['N']:
                    n_agent = agents[n_dirs['N']]
                    filtered_upstream[0] += n_agent.stats.get('in_s', 0)
                    if n_agent.current_phase in [0, 2]: neighbor_phases[0] = 1.0
                if n_dirs['S']:
                    s_agent = agents[n_dirs['S']]
                    filtered_upstream[1] += s_agent.stats.get('in_n', 0)
                    if s_agent.current_phase in [0, 2]: neighbor_phases[1] = 1.0
                if n_dirs['E']:
                    e_agent = agents[n_dirs['E']]
                    filtered_upstream[2] += e_agent.stats.get('in_w', 0)
                    if e_agent.current_phase in [4, 6]: neighbor_phases[2] = 1.0
                if n_dirs['W']:
                    w_agent = agents[n_dirs['W']]
                    filtered_upstream[3] += w_agent.stats.get('in_e', 0)
                    if w_agent.current_phase in [4, 6]: neighbor_phases[3] = 1.0

                agent.step(filtered_upstream, neighbor_phases, eval_mode=True)

    traci.close()
    
    # Convert metrics to standard units (kg for CO2, L for Fuel)
    return {
        "delay": episode_delay,
        "teleports": episode_teleports,
        "co2_kg": episode_co2_mg / 1_000_000.0,
        "fuel_L": episode_fuel_mg / 750_000.0
    }

def print_and_write(file, text):
    print(text)
    file.write(text + "\n")

if __name__ == "__main__":
    
    # 1. Initialise RL Architecture
    print("Initializing RL Map and Extracting Weights...")
    traci.start(["sumo", "-c", MODELS["RL_Agent"]["cfg"]])
    tls_ids = traci.trafficlight.getIDList()
    neighbors = get_dynamic_neighbors(tls_ids)
    positions = {tls: traci.junction.getPosition(tls) for tls in tls_ids}

    model_path = f"{LOAD_DIR}/dqn_global.pth"
    agents = {}
    for tls_id in tls_ids:
        agents[tls_id] = DQNAgent(tls_id, YELLOW_DURATION, MIN_GREEN_DURATION, MAX_GREEN_DURATION)
        agents[tls_id].get_static_data()
        agents[tls_id].policy_net.load_state_dict(torch.load(model_path))
        agents[tls_id].policy_net.eval()
        agents[tls_id].epsilon = 0.0
        agents[tls_id].epsilon_min = 0.0

    traci.close()

    neighbor_map = {tls: {'N': None, 'S': None, 'E': None, 'W': None} for tls in tls_ids}
    for tls_id in tls_ids:
        ax, ay = positions[tls_id]
        for n_id in neighbors[tls_id]:
            nx, ny = positions[n_id]
            if ny > ay + 10:   neighbor_map[tls_id]['N'] = n_id
            elif ny < ay - 10: neighbor_map[tls_id]['S'] = n_id
            elif nx > ax + 10: neighbor_map[tls_id]['E'] = n_id
            elif nx < ax - 10: neighbor_map[tls_id]['W'] = n_id

    # 2. Run evaluations
    results = {}
    for model_name, info in MODELS.items():
        print(f"\n================ STARTING {model_name.upper()} ================")
        model_metrics = {"delay": [], "teleports": [], "co2_kg": [], "fuel_L": []}
        
        for i, seed in enumerate(SEEDS):
            if info["is_rl"]:
                metrics = run_episode(model_name, seed, agents, neighbor_map)
            else:
                metrics = run_episode(model_name, seed)
                
            model_metrics["delay"].append(metrics["delay"])
            model_metrics["teleports"].append(metrics["teleports"])
            model_metrics["co2_kg"].append(metrics["co2_kg"])
            model_metrics["fuel_L"].append(metrics["fuel_L"])
            
            print(f" >> Ep {i+1} (Seed {seed}) | Delay: {metrics['delay']:.0f}s | Teleports: {metrics['teleports']} | CO2: {metrics['co2_kg']:.1f}kg | Fuel: {metrics['fuel_L']:.1f}L")
            
        # Calculate Averages
        results[model_name] = {k: sum(v)/len(v) for k, v in model_metrics.items()}

    # 3. Generate Report
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_filename = f"../outputs/evaluation_reports/evaluation_report_{ABBR}_{timestamp}.txt"
    
    with open(report_filename, "w") as f:
        print_and_write(f, "=" * 70)
        print_and_write(f, f"  TRAFFIC CONTROL EVALUATION REPORT ({ABBR})")
        print_and_write(f, "=" * 70)
        print_and_write(f, f"Episodes Evaluated : {EPISODES}")
        print_and_write(f, f"Steps per Episode  : {STEPS_PER_EPISODE}")
        print_and_write(f, f"Seeds Used         : {SEEDS}")
        print_and_write(f, "\n--- ABSOLUTE METRICS (Averages over 10 episodes) ---")
        print_and_write(f, f"{'Model':<12} | {'Delay (s)':<12} | {'Teleports':<10} | {'CO2 (kg)':<10} | {'Fuel (L)':<10}")
        print_and_write(f, "-" * 70)
        
        for name, m in results.items():
            print_and_write(f, f"{name:<12} | {m['delay']:<12.0f} | {m['teleports']:<10.1f} | {m['co2_kg']:<10.1f} | {m['fuel_L']:<10.1f}")

        # Helper function for relative differences
        def rel_diff(val, base):
            if base == 0: return "N/A"
            diff = ((val - base) / base) * 100
            return f"{'+' if diff > 0 else ''}{diff:.2f}%"

        # Compare against Fixed
        if "Fixed" in results:
            print_and_write(f, "\n--- RELATIVE COMPARISON (vs. Fixed Time) ---")
            print_and_write(f, f"{'Model':<12} | {'Delay':<12} | {'Teleports':<10} | {'CO2':<10} | {'Fuel':<10}")
            print_and_write(f, "-" * 70)
            base = results["Fixed"]
            compare = results["RL_Agent"]
            print_and_write(f, f"{'RL Agent':<12} | "
                                f"{rel_diff(compare['delay'], base['delay']):<12} | "
                                f"{rel_diff(compare['teleports'], base['teleports']):<10} | "
                                f"{rel_diff(compare['co2_kg'], base['co2_kg']):<10} | "
                                f"{rel_diff(compare['fuel_L'], base['fuel_L']):<10}")

        # Compare against Actuated
        if "Actuated" in results:
            print_and_write(f, "\n--- RELATIVE COMPARISON (vs. Actuated) ---")
            print_and_write(f, f"{'Model':<12} | {'Delay':<12} | {'Teleports':<10} | {'CO2':<10} | {'Fuel':<10}")
            print_and_write(f, "-" * 70)
            base = results["Actuated"]
            compare = results["RL_Agent"]
            print_and_write(f, f"{'RL Agent':<12} | "
                                f"{rel_diff(compare['delay'], base['delay']):<12} | "
                                f"{rel_diff(compare['teleports'], base['teleports']):<10} | "
                                f"{rel_diff(compare['co2_kg'], base['co2_kg']):<10} | "
                                f"{rel_diff(compare['fuel_L'], base['fuel_L']):<10}")

    print(f"\nReport successfully saved to: {report_filename}")