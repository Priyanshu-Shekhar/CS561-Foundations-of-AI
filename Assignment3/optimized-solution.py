import numpy as np
from collections import defaultdict

class POMDP:
    def __init__(self):
        self.states = set()
        self.observations = set()
        self.actions = set()
        self.initial_probability = {}
        self.transition_probability = {}
        self.observation_probability = {}
        self.default_weight = 0

    def parse_state_weights(self, filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # First line should be "state_weights"
            if not lines[0] == "state_weights":
                raise ValueError("Invalid format: First line should be 'state_weights'")
            
            # Parse number of states and default weight
            parts = lines[1].split()
            num_states = int(parts[0])
            self.default_weight = float(parts[1]) if len(parts) > 1 else 0
            
            # Calculate total weight for normalization
            total_weight = 0
            for i in range(2, len(lines)):
                state, weight = lines[i].replace('"', '').split()
                total_weight += float(weight)
            
            # Normalize probabilities
            for i in range(2, len(lines)):
                state, weight = lines[i].replace('"', '').split()
                self.states.add(state)
                self.initial_probability[state] = float(weight) / total_weight if total_weight > 0 else 1.0 / num_states

    def parse_state_observation_weights(self, filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if not lines[0] == "state_observation_weights":
                raise ValueError("Invalid format: First line should be 'state_observation_weights'")
            
            num_pairs, num_states, num_obs, default = map(int, lines[1].split())
            self.default_weight = default
            
            # Initialize with default weights
            self.observation_probability = defaultdict(lambda: defaultdict(lambda: float(default)))
            
            # Parse and store weights
            state_totals = defaultdict(float)
            weights = defaultdict(lambda: defaultdict(float))
            
            for line in lines[2:]:
                state, obs, weight = line.replace('"', '').split()
                self.states.add(state)
                self.observations.add(obs)
                weights[state][obs] = float(weight)
                state_totals[state] += float(weight)
            
            # Normalize probabilities
            for state in weights:
                for obs in weights[state]:
                    if state_totals[state] > 0:
                        self.observation_probability[state][obs] = weights[state][obs] / state_totals[state]
                    else:
                        self.observation_probability[state][obs] = 1.0 / len(self.observations)

    def parse_state_action_state_weights(self, filename):
        with open(filename, 'r') as f:
            lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if not lines[0] == "state_action_state_weights":
                raise ValueError("Invalid format: First line should be 'state_action_state_weights'")
            
            num_triples, num_states, num_actions, default = map(int, lines[1].split())
            self.default_weight = default
            
            # Initialize transition probabilities
            self.transition_probability = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: float(default))))
            
            # Parse and accumulate weights
            state_action_totals = defaultdict(float)
            weights = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
            
            for line in lines[2:]:
                curr_state, action, next_state, weight = line.replace('"', '').split()
                self.states.add(curr_state)
                self.states.add(next_state)
                self.actions.add(action)
                weights[curr_state][action][next_state] = float(weight)
                state_action_totals[(curr_state, action)] += float(weight)
            
            # Normalize probabilities
            for curr_state in weights:
                for action in weights[curr_state]:
                    total = state_action_totals[(curr_state, action)]
                    if total > 0:
                        for next_state in weights[curr_state][action]:
                            self.transition_probability[curr_state][action][next_state] = (
                                weights[curr_state][action][next_state] / total
                            )
                    else:
                        # Uniform distribution if no transitions
                        for next_state in self.states:
                            self.transition_probability[curr_state][action][next_state] = 1.0 / len(self.states)

    def viterbi(self, observations, actions):
        T = len(observations)
        if T == 0:
            return []
        
        # Initialize with log probabilities
        V = [{} for _ in range(T)]
        backptr = [{} for _ in range(T)]
        
        # Initialize first step
        for s in self.states:
            init_prob = self.initial_probability.get(s, 1.0 / len(self.states))
            obs_prob = self.observation_probability[s].get(observations[0], self.default_weight)
            
            if init_prob > 0 and obs_prob > 0:
                V[0][s] = np.log(init_prob) + np.log(obs_prob)
            else:
                V[0][s] = float('-inf')
            backptr[0][s] = None
        
        # Forward pass
        for t in range(1, T):
            for curr_state in self.states:
                max_prob = float('-inf')
                best_prev = None
                obs_prob = self.observation_probability[curr_state].get(observations[t], self.default_weight)
                
                if obs_prob > 0:
                    for prev_state in self.states:
                        if V[t-1][prev_state] == float('-inf'):
                            continue
                        
                        trans_prob = self.transition_probability[prev_state][actions[t-1]].get(curr_state, self.default_weight)
                        if trans_prob <= 0:
                            continue
                        
                        prob = V[t-1][prev_state] + np.log(trans_prob) + np.log(obs_prob)
                        if prob > max_prob:
                            max_prob = prob
                            best_prev = prev_state
                
                V[t][curr_state] = max_prob
                backptr[t][curr_state] = best_prev
        
        # Backward pass
        best_prob = float('-inf')
        best_last = None
        
        for s in self.states:
            if V[T-1][s] > best_prob:
                best_prob = V[T-1][s]
                best_last = s
        
        if best_last is None:
            return list(self.states)[:T]  # Return default sequence if no valid path
        
        # Reconstruct path
        path = [best_last]
        for t in range(T-1, 0, -1):
            prev = backptr[t][path[0]]
            if prev is None:
                prev = list(self.states)[0]
            path.insert(0, prev)
        
        return path

def read_observation_actions(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        if not lines[0] == "observation_actions":
            raise ValueError("Invalid format: First line should be 'observation_actions'")
        
        num_pairs = int(lines[1])
        observations = []
        actions = []
        
        for line in lines[2:]:
            parts = line.replace('"', '').split()
            observations.append(parts[0])
            actions.append(parts[1] if len(parts) > 1 else "N")
        
        return observations, actions

def write_output(filename, states):
    with open(filename, 'w') as f:
        f.write("states\n")
        f.write(str(len(states)) + "\n")
        for state in states:
            f.write(f'"{state}"\n')

def main():
    try:
        pomdp = POMDP()
        pomdp.parse_state_weights("state_weights.txt")
        pomdp.parse_state_observation_weights("state_observation_weights.txt")
        pomdp.parse_state_action_state_weights("state_action_state_weights.txt")
        observations, actions = read_observation_actions("observation_actions.txt")
        state_sequence = pomdp.viterbi(observations, actions)
        write_output("states.txt", state_sequence)
    except Exception as e:
        print(f"Error: {e}")
        # Write default output in case of error
        write_output("states.txt", ["S0"] * 1)

if __name__ == "__main__":
    main()