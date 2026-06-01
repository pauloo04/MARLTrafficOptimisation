import os
import subprocess
import xml.etree.ElementTree as ET

NET_FILE = "simple.net.xml"
ROU_FILE = "simple.rou.xml"
GRID_N = 6
GRID_LEN = 500
TOTAL_LEN = GRID_LEN * (GRID_N - 1)

print(f"--- 1. generating 4x4 Base Grid ({TOTAL_LEN}x{TOTAL_LEN}m) ---")
# Generate the base grid (outer edges act as spawn/exit points, leaving a 4x4 core)
subprocess.run([
    "netgenerate", "--grid", f"--grid.number={GRID_N}", "--grid.length", str(GRID_LEN),
    "--default.lanenumber", "2", "--output-file", NET_FILE, "--no-turnarounds"
], check=True)

print("--- 2. Identifying Central Nodes ---")
tree = ET.parse(NET_FILE)
root = tree.getroot()

tls_nodes = []
# Grab only the inner 16 intersections for our agents
for junction in root.findall("junction"):
    jid = junction.get("id")
    if jid.startswith(":"): continue

    x = float(junction.get("x"))
    y = float(junction.get("y"))

    if 10 < x < (TOTAL_LEN - 10) and 10 < y < (TOTAL_LEN - 10):
        tls_nodes.append(jid)

tls_str = ",".join(tls_nodes)

print(f"--- 3. Re-building Network with 4 TLS ---")
# Re-bake the grid to attach traffic lights to the central nodes
subprocess.run([
    "netgenerate", "--grid", f"--grid.number={GRID_N}", "--grid.length", str(GRID_LEN),
    "--default.lanenumber", "2", "--tls.left-green.time", "15", "--output-file", NET_FILE, 
    "--tls.set", tls_str, "--no-turnarounds"
], check=True)

subprocess.run(["netconvert", "-s", NET_FILE, "--plain-output-prefix", "plain"], check=True)
orig_tree = ET.parse(NET_FILE)
orig_root = orig_tree.getroot()

with open("plain.con.xml", "w") as f:
    f.write('<connections>\n')
    for conn in orig_root.findall("connection"):
        if conn.get("from").startswith(":"): continue 
        
        from_lane = conn.get("fromLane")
        dir_val = conn.get("dir")
        
        # Enforce lane rules: right lane is straight/right, left lane is strictly protected lefts
        if conn.get("tl") is not None:
            if from_lane == "1" and dir_val != "l":
                continue 
                
            if from_lane == "0" and dir_val == "l":
                continue 
            
        f.write(f'    <connection from="{conn.get("from")}" to="{conn.get("to")}" fromLane="{from_lane}" toLane="{conn.get("toLane")}"/>\n')
    f.write('</connections>\n')

# Apply the fixed connections back to the network
subprocess.run([
    "netconvert", "-n", "plain.nod.xml", "-e", "plain.edg.xml", 
    "-x", "plain.con.xml", "-o", NET_FILE
], check=True)

# Clean up temp files
for ext in ["nod", "edg", "con", "tll", "typ"]:
    if os.path.exists(f"plain.{ext}.xml"):
        os.remove(f"plain.{ext}.xml")

if os.path.exists("plain.netccfg"):
    os.remove("plain.netccfg")

print("--- 4. Identifying Edges for Traffic Flows ---")

tree = ET.parse(NET_FILE)
root = tree.getroot()

routes = []

def find_edge(x_from, y_from, x_to, y_to):
    for edge in root.findall("edge"):
        if edge.get("function") == "internal": continue
        
        from_j = edge.get("from")
        to_j = edge.get("to")
        
        fj_node = root.find(f"./junction[@id='{from_j}']")
        tj_node = root.find(f"./junction[@id='{to_j}']")
        
        fx, fy = float(fj_node.get("x")), float(fj_node.get("y"))
        tx, ty = float(tj_node.get("x")), float(tj_node.get("y"))
        
        if (abs(fx-x_from) < 50 and abs(fy-y_from) < 50 and
            abs(tx-x_to) < 50 and abs(ty-y_to) < 50):
            return edge.get("id")
    return None

v_starts, v_ends = [], []
h_starts, h_ends = [], []

# Map outer edges for flow generation
for i in range(1, GRID_N - 1):
    coord = i * GRID_LEN
    v_starts.append(find_edge(coord, 0, coord, GRID_LEN))
    v_ends.append(find_edge(coord, TOTAL_LEN - GRID_LEN, coord, TOTAL_LEN))
    
    v_starts.append(find_edge(coord, TOTAL_LEN, coord, TOTAL_LEN - GRID_LEN))
    v_ends.append(find_edge(coord, GRID_LEN, coord, 0))

    h_starts.append(find_edge(0, coord, GRID_LEN, coord))
    h_ends.append(find_edge(TOTAL_LEN - GRID_LEN, coord, TOTAL_LEN, coord))

    h_starts.append(find_edge(TOTAL_LEN, coord, TOTAL_LEN - GRID_LEN, coord))
    h_ends.append(find_edge(GRID_LEN, coord, 0, coord))

print("--- 5. Generating Route File ---")

with open(ROU_FILE, "w") as f:
    f.write('<routes>\n')
    f.write('    <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89"/>\n\n')
    
    def write_phase(phase_num, begin, end, primary_starts, primary_ends, secondary_starts, secondary_ends, primary_prob, turn_prob, bg_prob):
        f.write(f'    \n')
        
        for i, start_edge in enumerate(primary_starts):
            f.write(f'    <flow id="p{phase_num}_{start_edge}_str" type="car" begin="{begin}" end="{end}" probability="{primary_prob}" from="{start_edge}" to="{primary_ends[i]}"/>\n')
            for end_edge in secondary_ends:
                f.write(f'    <flow id="p{phase_num}_{start_edge}_turn_{end_edge}" type="car" begin="{begin}" end="{end}" probability="{turn_prob}" from="{start_edge}" to="{end_edge}"/>\n')
        
        for i, start_edge in enumerate(secondary_starts):
            f.write(f'    <flow id="p{phase_num}_{start_edge}_bg_str" type="car" begin="{begin}" end="{end}" probability="{bg_prob}" from="{start_edge}" to="{secondary_ends[i]}"/>\n')
            for end_edge in primary_ends:
                f.write(f'    <flow id="p{phase_num}_{start_edge}_bg_turn_{end_edge}" type="car" begin="{begin}" end="{end}" probability="{bg_prob/2:.3f}" from="{start_edge}" to="{end_edge}"/>\n')
        f.write('\n')

    # Define traffic waves to simulate rush hours
    # Phase 1: Vertical Rush
    write_phase(1, 0, 1200, v_starts, v_ends, h_starts, h_ends, primary_prob=0.05, turn_prob=0.02, bg_prob=0.01)
    
    # Phase 2: Horizontal Rush
    write_phase(2, 1200, 2400, h_starts, h_ends, v_starts, v_ends, primary_prob=0.05, turn_prob=0.02, bg_prob=0.01)
    
    # Phase 3: Chaos
    write_phase(3, 2400, 3600, v_starts, v_ends, h_starts, h_ends, primary_prob=0.04, turn_prob=0.025, bg_prob=0.04)

    f.write('</routes>\n')

print("--- 6. Generating TLS Logic File ---")

# Setup the fixed-time and actuated baselines for our evaluation runs
with open("tls_fixed.add.xml", "w") as f:
    f.write('<additional>\n')
    for t_id in tls_nodes:
        f.write(f'    <tlLogic id="{t_id}" type="static" programID="static_baseline" offset="0">\n')
        f.write('        <phase duration="31" state="GGrrrrGGrrrr"/>\n')
        f.write('        <phase duration="3"  state="yyrrrryyrrrr"/>\n')
        f.write('        <phase duration="10" state="rrGrrrrrGrrr"/>\n')
        f.write('        <phase duration="3"  state="rryrrrrryrrr"/>\n')
        f.write('        <phase duration="31" state="rrrGGrrrrGGr"/>\n')
        f.write('        <phase duration="3"  state="rrryyrrrryyr"/>\n')
        f.write('        <phase duration="10" state="rrrrrGrrrrrG"/>\n')
        f.write('        <phase duration="3"  state="rrrrryrrrrry"/>\n')
        f.write('    </tlLogic>\n')
    f.write('</additional>\n')

with open("tls_actuated.add.xml", "w") as f:
    f.write('<additional>\n')
    for t_id in tls_nodes:
        f.write(f'    <tlLogic id="{t_id}" type="actuated" programID="actuated_baseline" offset="0">\n')
        f.write('        <phase duration="31" minDur="5" maxDur="40" state="GGrrrrGGrrrr"/>\n')
        f.write('        <phase duration="3"  state="yyrrrryyrrrr"/>\n')
        f.write('        <phase duration="10" minDur="5" maxDur="40" state="rrGrrrrrGrrr"/>\n')
        f.write('        <phase duration="3"  state="rryrrrrryrrr"/>\n')
        f.write('        <phase duration="31" minDur="5" maxDur="40" state="rrrGGrrrrGGr"/>\n')
        f.write('        <phase duration="3"  state="rrryyrrrryyr"/>\n')
        f.write('        <phase duration="10" minDur="5" maxDur="40" state="rrrrrGrrrrrG"/>\n')
        f.write('        <phase duration="3"  state="rrrrryrrrrry"/>\n')
        f.write('    </tlLogic>\n')
    f.write('</additional>\n')

# Setup a dummy static logic for the RL agent (effectively an infinite sandbox)
with open("tls_rl.add.xml", "w") as f:
    f.write('<additional>\n')
    for t_id in tls_nodes:
        f.write(f'    <tlLogic id="{t_id}" type="static" programID="custom_agent_logic" offset="0">\n')
        f.write('        <phase duration="3600" state="GGrrrrGGrrrr"/>\n')
        f.write('        <phase duration="3600"  state="yyrrrryyrrrr"/>\n')
        f.write('        <phase duration="3600" state="rrGrrrrrGrrr"/>\n')
        f.write('        <phase duration="3600"  state="rryrrrrryrrr"/>\n')
        f.write('        <phase duration="3600" state="rrrGGrrrrGGr"/>\n')
        f.write('        <phase duration="3600"  state="rrryyrrrryyr"/>\n')
        f.write('        <phase duration="3600" state="rrrrrGrrrrrG"/>\n')
        f.write('        <phase duration="3600"  state="rrrrryrrrrry"/>\n')
        f.write('    </tlLogic>\n')

    f.write('\n  \n')

    # Inject E2 stop-bar detectors so the agents can observe queue lengths
    for edge in root.findall("edge"):
        if edge.get("function") == "internal": 
            continue
            
        to_node = edge.get("to")
        if to_node in tls_nodes:
            for lane in edge.findall("lane"):
                lane_id = lane.get("id")
                detector_id = f"e2_{lane_id}"
                
                f.write(f'    <laneAreaDetector id="{detector_id}" lane="{lane_id}" pos="-50" endPos="-1" freq="1" file="NUL"/>\n')
    f.write('</additional>\n')

print("--- 4x4 ENVIRONMENT READY (With Auto-Routing) ---")