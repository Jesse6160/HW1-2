from flask import Flask, render_template, request, jsonify
import random
import numpy as np

app = Flask(__name__)

# 儲存網格狀態
grid_state = {
    'size': 5,
    'start': None,
    'end': None,
    'obstacles': [],
    'policy': None,  # 儲存策略
    'values': None  # 儲存價值
}

# 行動對應的方向
ACTIONS = ['↑', '↓', '←', '→']
ACTION_MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右


def generate_random_policy(size):
    return np.random.choice(ACTIONS, size=(size, size)).tolist()


def value_iteration(state):
    size = state['size']
    values = np.zeros((size, size))
    policy = np.full((size, size), '↑', dtype='<U1')
    theta = 0.01  # 收斂閾值
    gamma = 0.9  # 折扣因子

    while True:
        delta = 0
        new_values = np.copy(values)

        for i in range(size):
            for j in range(size):
                if (i, j) == state['end']:
                    new_values[i][j] = 1  # 終點獎勵
                    continue
                if (i, j) in state['obstacles']:
                    new_values[i][j] = 0  # 障礙物無價值
                    continue

                max_value = float('-inf')
                best_action = '↑'
                for action_idx, (di, dj) in enumerate(ACTION_MOVES):
                    next_i, next_j = i + di, j + dj
                    reward = 0
                    if (next_i < 0 or next_i >= size or next_j < 0 or next_j >= size or
                            (next_i, next_j) in state['obstacles']):
                        reward = -0.1  # 撞牆或障礙物
                        next_i, next_j = i, j
                    elif (next_i, next_j) == state['end']:
                        reward = 1

                    q_value = reward + gamma * values[next_i][next_j]
                    if q_value > max_value:
                        max_value = q_value
                        best_action = ACTIONS[action_idx]

                new_values[i][j] = max_value
                policy[i][j] = best_action
                delta = max(delta, abs(new_values[i][j] - values[i][j]))

        values = new_values
        if delta < theta:
            break

    return values.tolist(), policy.tolist()


def policy_iteration(state):
    size = state['size']
    policy = np.array(generate_random_policy(size))
    values = np.zeros((size, size))
    gamma = 0.9
    theta = 0.01

    while True:
        # 策略評估
        while True:
            delta = 0
            new_values = np.copy(values)
            for i in range(size):
                for j in range(size):
                    if (i, j) == state['end']:
                        new_values[i][j] = 1
                        continue
                    if (i, j) in state['obstacles']:
                        new_values[i][j] = 0
                        continue

                    action_idx = ACTIONS.index(policy[i][j])
                    di, dj = ACTION_MOVES[action_idx]
                    next_i, next_j = i + di, j + dj
                    reward = 0
                    if (next_i < 0 or next_i >= size or next_j < 0 or next_j >= size or
                            (next_i, next_j) in state['obstacles']):
                        reward = -0.1
                        next_i, next_j = i, j
                    elif (next_i, next_j) == state['end']:
                        reward = 1

                    new_values[i][j] = reward + gamma * values[next_i][next_j]
                    delta = max(delta, abs(new_values[i][j] - values[i][j]))

            values = new_values
            if delta < theta:
                break

        # 策略改進
        policy_stable = True
        for i in range(size):
            for j in range(size):
                if (i, j) == state['end'] or (i, j) in state['obstacles']:
                    continue

                old_action = policy[i][j]
                max_value = float('-inf')
                best_action = '↑'
                for action_idx, (di, dj) in enumerate(ACTION_MOVES):
                    next_i, next_j = i + di, j + dj
                    reward = 0
                    if (next_i < 0 or next_i >= size or next_j < 0 or next_j >= size or
                            (next_i, next_j) in state['obstacles']):
                        reward = -0.1
                        next_i, next_j = i, j
                    elif (next_i, next_j) == state['end']:
                        reward = 1

                    q_value = reward + gamma * values[next_i][next_j]
                    if q_value > max_value:
                        max_value = q_value
                        best_action = ACTIONS[action_idx]

                policy[i][j] = best_action
                if old_action != best_action:
                    policy_stable = False

        if policy_stable:
            break

    return values.tolist(), policy.tolist()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/init_grid', methods=['POST'])
def init_grid():
    global grid_state
    size = int(request.json['size'])
    if 5 <= size <= 9:
        grid_state = {
            'size': size,
            'start': None,
            'end': None,
            'obstacles': [],
            'policy': generate_random_policy(size),
            'values': None
        }
        return jsonify({'success': True, 'size': size, 'policy': grid_state['policy']})
    return jsonify({'success': False, 'message': 'Size must be between 5 and 9'})


@app.route('/update_cell', methods=['POST'])
def update_cell():
    global grid_state
    data = request.json
    x, y = data['x'], data['y']
    cell_type = data['type']

    if cell_type == 'start':
        grid_state['start'] = (x, y)
    elif cell_type == 'end':
        grid_state['end'] = (x, y)
    elif cell_type == 'obstacle':
        if len(grid_state['obstacles']) < grid_state['size'] - 2:
            grid_state['obstacles'].append((x, y))

    # 當起點和終點都設置後，執行優化
    if grid_state['start'] and grid_state['end']:
        # 可以選擇值迭代或策略迭代，這裡默認使用值迭代
        grid_state['values'], grid_state['policy'] = value_iteration(grid_state)
        # 若要使用策略迭代，取消註釋以下行：
        # grid_state['values'], grid_state['policy'] = policy_iteration(grid_state)

    return jsonify(grid_state)


if __name__ == '__main__':
    app.run(debug=True)