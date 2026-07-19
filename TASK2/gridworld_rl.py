"""
Value Iteration and Policy Iteration on a 5x5 GridWorld (from scratch, pure NumPy)
No external RL libraries used - self contained so it always runs.
"""
import numpy as np

# ---------- 1. Define the GridWorld ----------
GRID_SIZE = 5
GAMMA = 0.9          # discount factor
THETA = 1e-4         # convergence threshold

# Grid legend:  S = start, G = goal (+1 reward), H = hole (-1 reward, terminal), . = normal cell (-0.04 step cost)
grid_layout = [
    ['S', '.', '.', '.', '.'],
    ['.', 'H', '.', 'H', '.'],
    ['.', '.', '.', '.', '.'],
    ['.', 'H', '.', '.', '.'],
    ['.', '.', '.', 'H', 'G'],
]

ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_DELTA = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1),
}

def is_terminal(r, c):
    return grid_layout[r][c] in ('H', 'G')

def reward(r, c):
    cell = grid_layout[r][c]
    if cell == 'G':
        return 1.0
    if cell == 'H':
        return -1.0
    return -0.04  # small step cost so the agent prefers shorter paths

def next_state(r, c, action):
    """Deterministic transition. Bumping into a wall keeps the agent in place."""
    if is_terminal(r, c):
        return r, c
    dr, dc = ACTION_DELTA[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
        return nr, nc
    return r, c  # wall bump

# ---------- 2. Value Iteration ----------
def value_iteration():
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    iteration = 0
    while True:
        delta = 0.0
        new_V = V.copy()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if is_terminal(r, c):
                    new_V[r, c] = reward(r, c)
                    continue
                action_values = []
                for a in ACTIONS:
                    nr, nc = next_state(r, c, a)
                    q_sa = reward(r, c) + GAMMA * V[nr, nc]
                    action_values.append(q_sa)
                new_V[r, c] = max(action_values)
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        V = new_V
        iteration += 1
        if delta < THETA:
            break
    return V, iteration

def extract_policy(V):
    policy = np.full((GRID_SIZE, GRID_SIZE), ' ', dtype='<U5')
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if is_terminal(r, c):
                policy[r, c] = grid_layout[r][c]
                continue
            best_a, best_val = None, -np.inf
            for a in ACTIONS:
                nr, nc = next_state(r, c, a)
                q_sa = reward(r, c) + GAMMA * V[nr, nc]
                if q_sa > best_val:
                    best_val = q_sa
                    best_a = a
            policy[r, c] = best_a
    return policy


# ---------- 3. Policy Iteration ----------
def policy_evaluation(policy, V):
    """Run until the value table stops changing much for the CURRENT fixed policy."""
    eval_sweeps = 0
    while True:
        delta = 0.0
        new_V = V.copy()
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if is_terminal(r, c):
                    new_V[r, c] = reward(r, c)
                    continue
                a = policy[r, c]
                nr, nc = next_state(r, c, a)
                new_V[r, c] = reward(r, c) + GAMMA * V[nr, nc]
                delta = max(delta, abs(new_V[r, c] - V[r, c]))
        V = new_V
        eval_sweeps += 1
        if delta < THETA:
            break
    return V, eval_sweeps

def policy_improvement(V):
    """Given a value table, greedily pick the best action in every state."""
    new_policy = np.full((GRID_SIZE, GRID_SIZE), 'UP', dtype='<U5')
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if is_terminal(r, c):
                new_policy[r, c] = grid_layout[r][c]
                continue
            best_a, best_val = None, -np.inf
            for a in ACTIONS:
                nr, nc = next_state(r, c, a)
                q_sa = reward(r, c) + GAMMA * V[nr, nc]
                if q_sa > best_val:
                    best_val = q_sa
                    best_a = a
            new_policy[r, c] = best_a
    return new_policy

def policy_iteration():
    V = np.zeros((GRID_SIZE, GRID_SIZE))
    # start with an arbitrary policy: always move RIGHT
    policy = np.full((GRID_SIZE, GRID_SIZE), 'RIGHT', dtype='<U5')
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if is_terminal(r, c):
                policy[r, c] = grid_layout[r][c]

    policy_improvement_steps = 0
    total_eval_sweeps = 0
    while True:
        V, sweeps = policy_evaluation(policy, V)
        total_eval_sweeps += sweeps
        new_policy = policy_improvement(V)
        policy_improvement_steps += 1
        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return V, policy, policy_improvement_steps, total_eval_sweeps

if __name__ == "__main__":
    print("================ VALUE ITERATION ================")
    V_val, num_iterations = value_iteration()
    policy_val = extract_policy(V_val)

    print(f"Converged in {num_iterations} iterations\n")
    print("Final Value Table (rounded to 2 decimals):")
    print(np.round(V_val, 2))
    print("\nExtracted Optimal Policy:")
    for row in policy_val:
        print(' '.join(f"{cell:>5}" for cell in row))

    print("\n\n================ POLICY ITERATION ===============")
    V_pol, policy_pol, improve_steps, eval_sweeps = policy_iteration()
    print(f"Policy Iteration converged after {improve_steps} policy-improvement rounds")
    print(f"(using {eval_sweeps} total policy-evaluation sweeps internally)\n")
    print("Final Value Table (rounded to 2 decimals):")
    print(np.round(V_pol, 2))
    print("\nFinal Optimal Policy:")
    for row in policy_pol:
        print(' '.join(f"{cell:>5}" for cell in row))
