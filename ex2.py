import itertools
import numpy as np
ids = ["111111111", "222222222"]
import random
import networkx as nx
from additional_inputs import additional_inputs
from inputs import small_inputs
import time
from copy import deepcopy
from collections import defaultdict

RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1

def initiate_agent(state):
    """
    initiate the agent with the given state
    """
    if state['optimal']:
        return OptimalTaxiAgent(state)
    return TaxiAgent(state)
class OptimalTaxiAgent:
    def __init__(self, initial):
        del initial['optimal']
        del initial["turns to go"]
        self.initial = initial
        self.initial['index'] = 0
        self.initial['score'] = 0
        self.map_old=initial['map']
        self.initial['graph'] = build_graph(self.initial['map'])
        self.initial['initial_state'] = deepcopy(initial)
        self.states=[]
        self.states.append(self.initial)
        locations=[(i, j) for i in range(len(initial['map'])) for j in range(len(initial['map'][0])) if initial['map'][i][j] != 'I']
        #measure how much time create all actions takes
        start_time = time.time()
        self.actions = create_all_actions(self.initial,locations)
        end_time=time.time()
        print("create all actions time: ",end_time-start_time)
        start_time = time.time()
        self.states, self.num_of_states = create_all_states(self.initial, self.actions)
        end_time=time.time()
        print("create all states time: ",end_time-start_time)
        self.pi = self.value_iteration()
        self.num_of_states+=1
    def value_iteration(self):
        """
        This function performs value iteration on the given state
        """
        # Initialize Markov Decision Process model
        actions = self.actions
        states = self.states
        rewards = lambda state: state['score']
        gamma = 0.9  # discount factor
        # Transition probabilities per state-action pair

        # Set value iteration parameters
        max_iter = 101  # Maximum number of iterations
        V = [0] * self.num_of_states  # Initialize value function
        pi =  [None] * self.num_of_states  # Initialize policy function

        # Start value iteration
        for i in range(1, max_iter):
            max_diff = 0  # Initialize max difference
            V_new =  [0] * self.num_of_states  # Initialize state values
            for s in set([s for s,s_next,p in states[i]]):
                max_val = 0
                for a in actions:
                    # Compute state value
                    val = rewards(s)  # Get direct reward
                    # take only the states that s is in their tuple
                    for s_next, p in [(s_next, p) for s_cur,s_next,p in states[i+1] if s_cur == s]:
                        val += states[i]*gamma * V[s_next['index']]  # Add discounted downstream values

                    # Store value best action so far
                    max_val = max(max_val, val)

                    # Update best policy
                    if V[s['index']] < val:
                        s_simple=simplify_state(s)
                        if a=='reset' and V[s['index']]==0:
                            pi[s['index']] = a
                        pi[str(s_simple)] = a  # Store action with highest value


                V_new[s['index']] = max_val  # Update value with highest value


            # Update value functions
            V = V_new

        return pi
    def act(self, state):
        return self.pi[str(state)]

def simplify_state(state):
    s=deepcopy(state)
    s['map']=state['map']
    s.pop('initial_state')
    s.pop('index')
    s.pop('score')
    return s


def probability_to_state(state, new_state):
    """
    calculate the probability of the state
    """
    prob=1
    for passenger in state['passengers']:
        prob *= probability_passenger_to_goal(state, passenger, new_state['passengers'][passenger]['destination'])
    return 0


def probability_passenger_to_goal(state, passenger, goal):
    if goal not in state['passengers'][passenger]['possible_goals']:
        return 1-state['passengers'][passenger]['prob_change_goal']
    if goal == state['passengers'][passenger]['destination']:
        return (1 - state['passengers'][passenger]['prob_change_goal']) + state['passengers'][passenger]['prob_change_goal']/len(state['passengers'][passenger]['possible_goals'])
    return state['passengers'][passenger]['prob_change_goal']/len(state['passengers'][passenger]['possible_goals'])


def create_all_actions(initial,locations):
    actions = {}
    for taxi in initial['taxis']:
        actions[taxi] = []
        for passenger in initial['passengers']:
            actions[taxi].append(('drop off', taxi, passenger))
            actions[taxi].append(('pick up', taxi, passenger))
        actions[taxi].append(('refuel', taxi))
        actions[taxi].append(('wait', taxi))
        for location in locations:
            actions[taxi].append(('move', taxi, location))
    # create
    actions = list(itertools.product(*actions.values()))
    actions.append('reset')
    # actions.append('terminate')
    return actions

def all_passengers_desti(initial):
    # create a dict with all the possible destinations for each passenger
    passenger_dest = dict()
    for passenger in initial['passengers']:
        passenger_dest[passenger] = initial['passengers'][passenger]['possible_goals']

    # Get all combinations of the elements in the lists
    combinations = list(itertools.product(*passenger_dest.values()))

    # Create a list of dictionaries by combining each combination with the original dictionary's keys
    dict_permutations = [dict(zip(passenger_dest.keys(), c)) for c in combinations]

    return dict_permutations

def create_all_states(initial, actions):
    states = []
    index=1
    graph=initial['graph']
    map=initial['map']
    del initial['graph']
    del initial['initial_state']
    del initial['index']
    states.append([(None, initial, 1)])
    dict_of_pass = all_passengers_desti(initial)
    for i, step in enumerate(states):
        print("step: ",i)
        if i ==101:
            break
        states.append([])
        for _,state,_ in step:
            for act in actions:
                if is_action_legal(state, act):
                    new_state = result(state, act)
                    print(new_state)
                    if any(t[1]==new_state for t in states[i+1]):
                        continue
                    # detrmnistic state
                    if act == 'reset' or act == 'terminate':
                        states[i + 1].append((state,new_state, 1))
                        continue
                    for permou in dict_of_pass:

                        for passenger in permou:
                            new_state=deepcopy(new_state)
                            new_state['passengers'][passenger]['destination'] = permou[passenger]
                        if new_state in [s for _, s, _ in states[i + 1]]:
                            continue
                        states[i + 1].append((state,new_state, probability_to_state(state, new_state)))
        print("num_of_states", len(step))
    return states, index
def drop_index(state):
    state=deepcopy(state)
    if 'index' in state.keys():
        state.pop('index')
    state.pop('graph')
    state.pop('initial_state')
    return state
class TaxiAgent:
 def __init__(self, initial):
     self.initial = initial

 def act(self, state):
     raise NotImplemented
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