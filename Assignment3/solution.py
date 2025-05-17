
import numpy as np

class POMDP:
    def __init__(self):
        # Initialize dictionaries to store probabilities
        self.states = set()
        self.observations = set()
        self.actions = set()
        
        self.initial_prob = {}
        self.trans_prob = {}
        self.obs_prob = {}
        
        # Small default weight to prevent log(0)
        self.default_weight = 1e-10

    def parse_state_weights(self, filename):
        try:
            with open(filename, 'r') as f:
                # Skip first line indicating file type
                lines = f.readlines()
                if not lines:
                    raise ValueError("Empty state weights file")

                # Parse the header line to get number of states and default weight
                header_parts = lines[1].strip().split()
                num_states = int(header_parts[0])
                # Default weight is optional, use small value if not present
                self.default_weight = float(header_parts[1]) if len(header_parts) > 1 else 1e-10

                weights = {}
                total_weight = 0
                
                # Parse weights for each state
                for line in lines[2:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Carefully handle lines that might have different formats
                    parts = line.replace('"', '').split()
                    
                    # Ensure we have at least 2 parts
                    if len(parts) >= 2:
                        state = parts[0]
                        try:
                            weight = float(parts[1])
                            self.states.add(state)
                            weights[state] = weight
                            total_weight += weight
                        except ValueError:
                            # Skip lines with invalid weight
                            continue
                
                # Normalize probabilities
                if total_weight > 0:
                    for state, weight in weights.items():
                        self.initial_prob[state] = weight / total_weight
                else:
                    # Uniform distribution if no weights or all zero
                    for state in self.states:
                        self.initial_prob[state] = 1.0 / max(1, len(self.states))

        except FileNotFoundError:
            # Fallback if file not found
            print(f"Warning: {filename} not found. Using uniform distribution.")
            if not self.states:
                self.states.add("S0")  # Add a default state
            self.initial_prob = {state: 1.0 / len(self.states) for state in self.states}
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            # Ensure we have at least one state
            if not self.states:
                self.states.add("S0")
            self.initial_prob = {state: 1.0 / len(self.states) for state in self.states}

    def parse_state_observation_weights(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                if not lines:
                    raise ValueError("Empty state observation weights file")

                # Parse header line
                header_parts = lines[1].strip().split()
                num_pairs = int(header_parts[0])
                num_states = int(header_parts[1])
                num_obs = int(header_parts[2])
                # Default weight is optional
                self.default_weight = float(header_parts[3]) if len(header_parts) > 3 else 1e-10

                obs_weights = {}
                state_totals = {}
                
                # Parse observation weights
                for line in lines[2:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.replace('"', '').split()
                    
                    # Ensure we have at least 3 parts
                    if len(parts) >= 3:
                        state, obs, weight = parts[:3]
                        
                        try:
                            weight = float(weight)
                            self.states.add(state)
                            self.observations.add(obs)
                            
                            if state not in obs_weights:
                                obs_weights[state] = {}
                                state_totals[state] = 0
                            
                            obs_weights[state][obs] = weight
                            state_totals[state] += weight
                        except ValueError:
                            # Skip lines with invalid weight
                            continue
                
                # Normalize observation probabilities
                self.obs_prob = {}
                for state in obs_weights:
                    self.obs_prob[state] = {}
                    total = state_totals[state]
                    
                    for obs, weight in obs_weights[state].items():
                        self.obs_prob[state][obs] = (
                            weight / total if total > 0 else 1.0 / max(1, len(self.observations))
                        )

        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using default observation probabilities.")
            # Fallback with uniform distribution
            self.obs_prob = {
                state: {obs: 1.0 / max(1, len(self.observations)) 
                        for obs in self.observations} 
                for state in self.states
            }
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            # Fallback with uniform distribution
            self.obs_prob = {
                state: {obs: 1.0 / max(1, len(self.observations)) 
                        for obs in self.observations} 
                for state in self.states
            }

    def parse_state_action_state_weights(self, filename):
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                if not lines:
                    raise ValueError("Empty state action state weights file")

                # Parse header line
                header_parts = lines[1].strip().split()
                num_triples = int(header_parts[0])
                num_states = int(header_parts[1])
                num_actions = int(header_parts[2])
                # Default weight is optional
                self.default_weight = float(header_parts[3]) if len(header_parts) > 3 else 1e-10

                trans_weights = {}
                state_action_totals = {}
                
                # Parse transition weights
                for line in lines[2:]:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.replace('"', '').split()
                    
                    # Ensure we have at least 4 parts
                    if len(parts) >= 4:
                        curr_state, action, next_state, weight = parts[:4]
                        
                        try:
                            weight = float(weight)
                            self.states.add(curr_state)
                            self.states.add(next_state)
                            self.actions.add(action)
                            
                            # Initialize nested dictionaries
                            if curr_state not in trans_weights:
                                trans_weights[curr_state] = {}
                            if action not in trans_weights[curr_state]:
                                trans_weights[curr_state][action] = {}
                            
                            # Track weights and totals
                            trans_weights[curr_state][action][next_state] = weight
                            
                            key = (curr_state, action)
                            state_action_totals[key] = state_action_totals.get(key, 0) + weight
                        except ValueError:
                            # Skip lines with invalid weight
                            continue
                
                # Normalize transition probabilities
                self.trans_prob = {}
                for curr_state in trans_weights:
                    self.trans_prob[curr_state] = {}
                    for action in trans_weights[curr_state]:
                        total = state_action_totals.get((curr_state, action), 0)
                        self.trans_prob[curr_state][action] = {}
                        
                        for next_state in trans_weights[curr_state][action]:
                            weight = trans_weights[curr_state][action][next_state]
                            # Normalize or use uniform distribution
                            self.trans_prob[curr_state][action][next_state] = (
                                weight / total if total > 0 else 1.0 / max(1, len(self.states))
                            )

        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using default transition probabilities.")
            # Fallback with uniform distribution
            self.trans_prob = {
                state1: {
                    action: {
                        state2: 1.0 / max(1, len(self.states)) 
                        for state2 in self.states
                    } 
                    for action in self.actions
                } 
                for state1 in self.states
            }
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            # Fallback with uniform distribution
            self.trans_prob = {
                state1: {
                    action: {
                        state2: 1.0 / max(1, len(self.states)) 
                        for state2 in self.states
                    } 
                    for action in self.actions
                } 
                for state1 in self.states
            }

    def viterbi(self, observations, actions):
        if not observations:
            return []

        # Logarithmic computation to avoid underflow
        def safe_log(x):
            # Add a very small epsilon to avoid log(0)
            return np.log(max(x, 1e-300))

        # Initialize dynamic programming tables
        T = len(observations)
        V = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]

        # Initial probabilities
        for state in self.states:
            # Use default weight if key is missing
            init_prob = self.initial_prob.get(state, self.default_weight)
            obs_prob = self.obs_prob.get(state, {}).get(observations[0], self.default_weight)
            V[0][state] = safe_log(init_prob) + safe_log(obs_prob)
            backpointer[0][state] = state  # Point to self for initial state

        # Dynamic programming
        for t in range(1, T):
            for curr_state in self.states:
                obs_prob = self.obs_prob.get(curr_state, {}).get(observations[t], self.default_weight)
                max_prob = float('-inf')
                best_prev_state = None

                for prev_state in self.states:
                    # Use get with default instead of direct access
                    trans_prob = (self.trans_prob
                        .get(prev_state, {})
                        .get(actions[t-1], {})
                        .get(curr_state, self.default_weight)
                    )
                    
                    # Compute total probability
                    prob = (V[t-1][prev_state] + 
                            safe_log(trans_prob) + 
                            safe_log(obs_prob))

                    # Update best path
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_state = prev_state

                V[t][curr_state] = max_prob
                backpointer[t][curr_state] = best_prev_state

        # Backtrack to find best path
        best_last_state = max(V[-1], key=V[-1].get)
        path = [best_last_state]

        for t in range(T-1, 0, -1):
            # Safely get the previous state, defaulting to the current state
            prev_state = backpointer[t].get(path[0], path[0])
            path.insert(0, prev_state)

        return path

def read_observation_actions(filename):
    try:
        with open(filename, 'r') as f:
            # Skip first line (header)
            lines = f.readlines()[1:]
            
            observations = []
            actions = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.replace('"', '').split()
                
                if len(parts) == 1:
                    observations.append(parts[0])
                    actions.append("N")  # Default null action
                elif len(parts) >= 2:
                    observations.append(parts[0])
                    actions.append(parts[1])
            
            return observations, actions
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
        return [], []

def write_output(filename, states):
    try:
        with open(filename, 'w') as f:
            f.write("states\n")
            f.write(f"{len(states)}\n")
            for state in states:
                f.write(f'"{state}"\n')
    except Exception as e:
        print(f"Error writing output: {e}")

def main():
    # Create POMDP object
    pomdp = POMDP()

    # Parse input files
    pomdp.parse_state_weights("state_weights.txt")
    pomdp.parse_state_observation_weights("state_observation_weights.txt")
    pomdp.parse_state_action_state_weights("state_action_state_weights.txt")

    # Read observations and actions
    observations, actions = read_observation_actions("observation_actions.txt")

    # Run Viterbi algorithm
    state_sequence = pomdp.viterbi(observations, actions)

    # Write output
    write_output("states.txt", state_sequence)
    
if __name__ == "__main__":
    main()