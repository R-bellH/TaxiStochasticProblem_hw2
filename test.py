def is_action_legal(state, action):
    """
    check if the action is legal
    """
    def _is_move_action_legal(move_action):
        taxi_name = move_action[1]
        if taxi_name not in state['taxis'].keys():
            return False
        if state['taxis'][taxi_name]['fuel'] == 0:
            return False
        l1 = state['taxis'][taxi_name]['location']
        l2 = move_action[2]
        # check if the location next to l1 in map
        return l2 in list(state['graph'].neighbors(l1))

    def _is_pick_up_action_legal(pick_up_action):
        taxi_name = pick_up_action[1]
        passenger_name = pick_up_action[2]
        # check same position
        if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['location']:
            return False
        # check taxi capacity
        if state['taxis'][taxi_name]['capacity'] <= 0:
            return False
        # check passenger is not in his destination
        if state['passengers'][passenger_name]['destination'] == state['passengers'][passenger_name]['location']:
            return False
        return True


    def _is_drop_action_legal(drop_action):
        taxi_name = drop_action[1]
        passenger_name = drop_action[2]
        # check same position
        if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['destination']:
            return False
        # check passenger is in the taxi
        if state['passengers'][passenger_name]['location'] != taxi_name:
            return False
        return True

    def _is_refuel_action_legal(refuel_action):
        """
        check if taxi in gas location
        """
        taxi_name = refuel_action[1]
        i, j = state['taxis'][taxi_name]['location']
        if state['map'][i][j] == 'G':
            return True
        else:
            return False

    def _is_action_mutex(global_action):
        assert type(global_action) == tuple, "global action must be a tuple"
        # one action per taxi
        if len(set([a[1] for a in global_action])) != len(global_action):
            return True
        # pick up the same person
        pick_actions = [a for a in global_action if a[0] == 'pick up']
        if len(pick_actions) > 1:
            passengers_to_pick = set([a[2] for a in pick_actions])
            if len(passengers_to_pick) != len(pick_actions):
                return True

        return False

    if action == "reset":
        return True
    if action == "terminate":
        return True
    if len(action) != len(state["taxis"].keys()):
        return False
    for atomic_action in action:
        # illegal move action
        if atomic_action[0] == 'move':
            if not _is_move_action_legal(atomic_action):
                return False
        # illegal pick action
        elif atomic_action[0] == 'pick up':
            if not _is_pick_up_action_legal(atomic_action):
                return False
        # illegal drop action
        elif atomic_action[0] == 'drop off':
            if not _is_drop_action_legal(atomic_action):
                return False
        # illegal refuel action
        elif atomic_action[0] == 'refuel':
            if not _is_refuel_action_legal(atomic_action):
                return False
        elif atomic_action[0] != 'wait':
            return False
    # check mutex action
    if _is_action_mutex(action):
        return False
    # check taxis collision
    if len(state['taxis']) > 1:
        taxis_location_dict = dict([(t, state['taxis'][t]['location']) for t in state['taxis'].keys()])
        move_actions = [a for a in action if a[0] == 'move']
        for move_action in move_actions:
            taxis_location_dict[move_action[1]] = move_action[2]
        if len(set(taxis_location_dict.values())) != len(taxis_location_dict):
            return False
    return True
def result(state, action):
    """"
    update the state according to the action
    """
    return apply(state, action)
def apply(state, action):
    """
    apply the action to the state
    """
    if action == "reset":
        return reset_environment(state)
    if action == "terminate":
        return terminate_execution(state)
    new_state= deepcopy(state)
    for atomic_action in action:
        new_state=apply_atomic_action(new_state, atomic_action)
    return new_state

def apply_atomic_action(state, atomic_action):
    """
    apply an atomic action to the state
    """
    state= deepcopy(state)
    taxi_name = atomic_action[1]
    if atomic_action[0] == 'move':
        state['taxis'][taxi_name]['location'] = atomic_action[2]
        state['taxis'][taxi_name]['fuel'] -= 1
        return state
    elif atomic_action[0] == 'pick up':
        passenger_name = atomic_action[2]
        state['taxis'][taxi_name]['capacity'] -= 1
        state['passengers'][passenger_name]['location'] = taxi_name
        return state
    elif atomic_action[0] == 'drop off':
        passenger_name = atomic_action[2]
        state['passengers'][passenger_name]['location'] = state['taxis'][taxi_name]['location']
        state['taxis'][taxi_name]['capacity'] += 1
        state['score'] += DROP_IN_DESTINATION_REWARD
        return state
    elif atomic_action[0] == 'refuel':
        state['taxis'][taxi_name]['fuel'] = state['initial_state']['taxis'][taxi_name]['fuel']
        state['score'] -= REFUEL_PENALTY
        return state
    elif atomic_action[0] == 'wait':
        return state
    else:
        raise NotImplemented

def reset_environment(state):
    """
    reset the state of the environment
    """
    state["taxis"] = deepcopy(state["initial_state"]["taxis"])
    state["passengers"] = deepcopy(state["initial_state"]["passengers"])
    state["score"] -= RESET_PENALTY
    return state

def terminate_execution(state):
    """
    terminate the execution of the problem
    """
    return 'EndOfGame'

class EndOfGame(Exception):
    """
    Exception to be raised when the game is over
    """
    pass
def build_graph(map):
    """
    build the graph of the problem
    """
    n, m = len(map), len(map[0])
    g = nx.grid_graph((m, n))
    nodes_to_remove = []
    for node in g:
        if map[node[0]][node[1]] == 'I':
            nodes_to_remove.append(node)
    for node in nodes_to_remove:
        g.remove_node(node)
    return g