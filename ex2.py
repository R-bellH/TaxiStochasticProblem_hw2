import itertools
import json
import time
from copy import deepcopy
from collections import defaultdict
import networkx as nx
import numpy as np

ids = ["111111111", "222222222"]
RESET_PENALTY = 50
REFUEL_PENALTY = 10
DROP_IN_DESTINATION_REWARD = 100
INIT_TIME_LIMIT = 300
TURN_TIME_LIMIT = 0.1

def encode(d: dict) -> str:
    return str(d)


def decode(s: str) -> dict:
    if s == 'EndOfGame':
        return s
    return eval(s)

class OptimalTaxiAgent:
    def __init__(self, initial):
        self.time_start=time.time()
        self.initial = initial
        self.max_turns = initial['turns to go']
        self.rewards=dict()
        self.next_state_probability=defaultdict(list)
        self.all_possible_passengers=self.all_passengers_desti()
        print('all possible passengers: done')
        self.actions = self.create_all_actions()
        print('Number of actions: ', len(self.actions))
        print('create all actions: done')
        self.graph = self.build_graph()
        self.states = self.create_all_states()
        for key, value in self.next_state_probability.items():
            new_value=[]
            for v in value:
                new_value.append((encode(v[0]),v[1]))
            self.next_state_probability[key]=set(new_value)
        print('create all states: done')
        self.num_of_states = 0
        self.indexs=dict()
     # index
        for i in range(1, len(self.states)):
            for j in range(len(self.states[i])):
                if encode(self.states[i][j][1]) not in self.indexs:
                    self.indexs[encode(self.states[i][j][1])]=self.num_of_states
                    self.num_of_states+=1
        print('index the states: done')
        print('number of states: ', self.num_of_states)
        self.policy, self.V = self.value_iteration()
        print('value iteration: done')
        print('time: ', time.time()-self.time_start)
    def value_iteration(self):
        """
        This function performs value iteration on the given state
        """
        # Initialize Markov Decision Process model
        actions = self.actions
        states = self.states
        rewards = lambda state: state['score']
        gamma = 1  # discount factor

        # Set value iteration parameters
        max_iter = self.max_turns # Maximum number of iterations
        V = dict()  # Initialize value function
        pi =  dict() # Initialize policy function

        # Start value iteration
        for i in range(0, max_iter):
            if time.time()- self.time_start > INIT_TIME_LIMIT-3:
                for j in range (i, max_iter):
                    pi[j] = ['terminate'] * self.num_of_states
                    V[j] =  [0] * self.num_of_states
                    return pi, V

            V_for_step_i =  [0] * self.num_of_states  # Initialize state values for step i
            pi_for_step_i = [None] * self.num_of_states  # Initialize policy for step i
            for old_state in set([encode(s_next) for s,s_next in states[self.max_turns-i]]):
                max_val = -51
                a_max = None
                for a in actions:
                    if self.next_state_probability[(old_state,a)] == []:
                        continue
                    current_v=self.rewards[a]
                    if i!=0:
                        for new_state, p in self.next_state_probability[(old_state, a)]:
                            current_v+=gamma*V[i-1][self.indexs[new_state]]*p
                    if current_v > max_val:
                        max_val = current_v
                        a_max = a

                pi_for_step_i[self.indexs[old_state]] = a_max
                V_for_step_i[self.indexs[old_state]] = max_val
            V[i] = V_for_step_i
            pi[i] = pi_for_step_i


        print('V init = ', V[self.max_turns-1][self.indexs[encode(self.initial)]])
        print("done")
        return pi , V



    def create_all_actions(self):
        actions = {}
        locations = [(i, j) for i in range(len(self.initial['map'])) for j in range(len(self.initial['map'][0])) if
                     self.initial['map'][i][j] != 'I']
        for taxi in self.initial['taxis']:
            actions[taxi] = []
            for passenger in self.initial['passengers']:
                actions[taxi].append(('drop off', taxi, passenger))
                actions[taxi].append(('pick up', taxi, passenger))
            actions[taxi].append(('refuel', taxi))
            actions[taxi].append(('wait', taxi))
            for location in locations:
                actions[taxi].append(('move', taxi, location))
        # create
        actions = list(itertools.product(*actions.values()))
        actions.append('reset')
        for a in actions:
            self.rewards[a] = self.reward(a)
        self.rewards[None]= 0
        return actions



    def create_all_states(self):
        states_after_i_steps=[[] for i in range(0,self.max_turns+1)]
        states_after_i_steps[0] =[(None, self.initial)]
        for i in range(0, self.max_turns):
            for step in states_after_i_steps[i]:
                old_state=step[1]
                for a in self.actions:
                    if self.is_action_legal(a,old_state):
                        new_state=self.apply(a,old_state)
                        if a=='reset' or a == 'terminate':
                            self.next_state_probability[(encode(old_state),a)].append((new_state,1))
                            if not self.state_exist(states_after_i_steps[i + 1], new_state):
                                states_after_i_steps[i + 1].append((old_state, new_state))
                        else:
                            self.next_state_probability[(encode(old_state), a)].append(
                                (new_state, self.probability_to_state(old_state, new_state)))
                            if not self.state_exist(states_after_i_steps[i+1],new_state):
                                states_after_i_steps[i + 1].append((old_state, new_state))
                            for permutation in self.all_possible_passengers:
                                newer_state = deepcopy(new_state)
                                for passen in permutation:
                                    newer_state['passengers'][passen]['destination']= permutation[passen]
                                self.next_state_probability[(encode(old_state), a)].append((newer_state,self.probability_to_state(old_state, newer_state)))
                                if not self.state_exist(states_after_i_steps[i + 1], newer_state):
                                    states_after_i_steps[i + 1].append((old_state, newer_state))
        return states_after_i_steps



    def probability_to_state(self,state, new_state):
        """
        calculate the probability of the state
        """
        prob = 1
        for passenger in state['passengers']:
            prob *= self.probability_passenger_to_goal(state, passenger, new_state['passengers'][passenger]['destination'])
        return prob

    def probability_passenger_to_goal(self, state, passenger, goal):
        if goal not in state['passengers'][passenger]['possible_goals']:
            return 1 - state['passengers'][passenger]['prob_change_goal']
        if goal == state['passengers'][passenger]['destination']:
            return (1 - state['passengers'][passenger]['prob_change_goal']) + state['passengers'][passenger][
                'prob_change_goal'] / len(state['passengers'][passenger]['possible_goals'])
        return state['passengers'][passenger]['prob_change_goal'] / len(
            state['passengers'][passenger]['possible_goals'])
    def all_passengers_desti(self):
        # create a dict with all the possible destinations for each passenger
        passenger_dest = dict()
        for passenger in self.initial['passengers']:
            passenger_dest[passenger] = self.initial['passengers'][passenger]['possible_goals']

        # Get all combinations of the elements in the lists
        combinations = list(itertools.product(*passenger_dest.values()))

        # Create a list of dictionaries by combining each combination with the original dictionary's keys
        dict_permutations = [dict(zip(passenger_dest.keys(), c)) for c in combinations]

        return dict_permutations

    def state_exist(self,states_after_i,new_state):
        return any([new_state==s[1] for s in states_after_i])
    def is_action_legal(self, action,state) -> bool:
        """
        check if the action is legal
        """
        def _is_move_action_legal(move_action, state):
            taxi_name = move_action[1]
            if taxi_name not in state['taxis'].keys():
                return False
            if state['taxis'][taxi_name]['fuel'] == 0:
                return False
            l1 = state['taxis'][taxi_name]['location']
            l2 = move_action[2]
            return l2 in list(self.graph.neighbors(l1))

        def _is_pick_up_action_legal(pick_up_action, state):
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

        def _is_drop_action_legal(drop_action, state):
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            # check same position
            if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['destination']:
                return False
            # check passenger is in the taxi
            if state['passengers'][passenger_name]['location'] != taxi_name:
                return False
            return True

        def _is_refuel_action_legal(refuel_action, state):
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
                if not _is_move_action_legal(atomic_action, state):
                    return False
            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action, state):
                    return False
            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action, state):
                    return False
            # illegal refuel action
            elif atomic_action[0] == 'refuel':
                if not _is_refuel_action_legal(atomic_action, state):
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
    def apply(self, action, state):
        """
        apply the action to the state
        """
        new_state=deepcopy(state)
        if action == "reset":
            new_state = self.reset_environment(new_state)
            return new_state
        if action == "terminate":
            self.terminate_execution(new_state)
            return new_state
        for atomic_action in action:
            new_state = self.apply_atomic_action(atomic_action, new_state)
        return new_state
    def apply_atomic_action(self, atomic_action, state):
        """
        apply an atomic action to the state
        """
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
            state['passengers'][passenger_name]['location'] =state['taxis'][taxi_name]['location']
            state['taxis'][taxi_name]['capacity'] += 1
            return state
        elif atomic_action[0] == 'refuel':
            state['taxis'][taxi_name]['fuel'] = self.initial['taxis'][taxi_name]['fuel']
            return state
        elif atomic_action[0] == 'wait':
            return state
        else:
            raise NotImplemented
    def reset_environment(self, state):
        """
        reset the state of the environment
        """
        state["taxis"] = self.initial["taxis"]
        state["passengers"] = self.initial["passengers"]
        return state

    def reward(self, action):
        reward=0
        if action == "reset":
            return -RESET_PENALTY
        if action == "terminate":
            return 0
        for atomic_action in action:
            if atomic_action[0] == 'drop off':
                reward += DROP_IN_DESTINATION_REWARD
            elif atomic_action[0] == 'refuel':
                reward -= REFUEL_PENALTY
        return reward


    def terminate_execution(self, state):
        """
        terminate the execution of the problem
        """
        return 'EndOfGame'
    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.initial['map']), len(self.initial['map'][0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.initial['map'][node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g
    def act(self, state):
        turns_to_go = state['turns to go']
        state['turns to go'] =self.max_turns
        action =self.policy[turns_to_go-1][self.indexs[encode(state)]]
        if action =='reset' and self.V[turns_to_go-1][self.indexs[encode(self.initial)]]<50:
            action='terminate'
        return action



class TaxiAgent:
    def __init__(self, initial):
        self.start_time= time.time()
        self.initial = initial
        self.graph=self.build_graph()
        self.depth_to_explore=10
        self.rewards = defaultdict(lambda: 0)
        self.actions = self.create_all_actions()
        self.Q=defaultdict(lambda: 0)
        self.Q = self.monte_carlo(initial, None, 0)
        while time.time()-self.start_time < 30:
            # give the best 5 states from Q
            best_Qs = sorted(self.Q, key=self.Q.get, reverse=True)[:5]
            # sample a Q from the best 5
            index = np.random.choice(range(len(best_Qs)))
            best_q = best_Qs[index]
            self.Q=self.monte_carlo(decode(best_q[0]), best_q[1], self.Q[best_q])






    def monte_carlo(self, initial, action, reward) -> defaultdict:
        """
        monte carlo simulation
        """
        eps=0.1
        alpha = 0.01
        Q=self.Q
        s,a,r = initial, action, reward
        expriances = [(s,a,r)]
        for _ in range(self.depth_to_explore):
            if s != 'EndOfGame':
                if np.random.rand() <eps:
                    a = np.random.choice(self.actions)
                    while not self.is_action_legal(a,s):
                        a = np.random.choice(self.actions)
                else:
                    a = self.best_action(encode(s),Q)
                s, r = self.apply(a, s), self.rewards[a]
                expriances.append((s,a,r))
            else:
                break
                # for t, exp in enumerate(expriances):
                #     s_t, a_t, r_t = exp
                #     if s_t not in [s for s,_,_ in expriances[:t]]:
                #         G = sum([r for _,_,r in expriances[t:]])
                #         Q[encode(s_t),a_t]+= alpha*(G-Q[encode(s_t),a_t])
                # s,a,r = initial, None,0
                # expriances = [(s,a,r)]
        for t, exp in enumerate(expriances):
            s_t, a_t, r_t = exp
            if s_t not in [s for s, _, _ in expriances[:t]]:
                G = sum([r for _, _, r in expriances[t:]])
                Q[encode(s_t), a_t] += alpha * (G - Q[encode(s_t), a_t])
        return Q


    def best_action(self,state,Q):
        if state=='EndOfGame':
            return None
        relevant_Q = [q for q in Q.keys() if q[0]==state and q[1]!=None]
        if state not in [q[0] for q in Q.keys()] or relevant_Q==[]:
            a = np.random.choice(self.actions)
            while not self.is_action_legal(a, decode(state)):
                a = np.random.choice(self.actions)
            return a
        best_action= max(relevant_Q, key=lambda x: Q[x])[1]
        while not self.is_action_legal(best_action, decode(state)):
            best_action = np.random.choice(self.actions)
        return best_action

    def create_all_actions(self):
            actions = {}
            locations = [(i, j) for i in range(len(self.initial['map'])) for j in range(len(self.initial['map'][0])) if
                         self.initial['map'][i][j] != 'I']
            for taxi in self.initial['taxis']:
                actions[taxi] = []
                for passenger in self.initial['passengers']:
                    actions[taxi].append(('drop off', taxi, passenger))
                    actions[taxi].append(('pick up', taxi, passenger))
                actions[taxi].append(('refuel', taxi))
                actions[taxi].append(('wait', taxi))
                for location in locations:
                    actions[taxi].append(('move', taxi, location))
            # create
            actions = list(itertools.product(*actions.values()))
            actions.append('reset')
            actions.append('terminate')
            for a in actions:
                self.rewards[a] = self.reward(a)
            return actions

    def is_action_legal(self, action, state) -> bool:
        """
        check if the action is legal
        """

        def _is_move_action_legal(move_action, state):
            taxi_name = move_action[1]
            if taxi_name not in state['taxis'].keys():
                return False
            if state['taxis'][taxi_name]['fuel'] == 0:
                return False
            l1 = state['taxis'][taxi_name]['location']
            l2 = move_action[2]
            return l2 in list(self.graph.neighbors(l1))

        def _is_pick_up_action_legal(pick_up_action, state):
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

        def _is_drop_action_legal(drop_action, state):
            taxi_name = drop_action[1]
            passenger_name = drop_action[2]
            # check same position
            if state['taxis'][taxi_name]['location'] != state['passengers'][passenger_name]['destination']:
                return False
            # check passenger is in the taxi
            if state['passengers'][passenger_name]['location'] != taxi_name:
                return False
            return True

        def _is_refuel_action_legal(refuel_action, state):
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
        if action == None:
            return False
        if len(action) != len(state["taxis"].keys()):
            return False
        for atomic_action in action:
            # illegal move action
            if atomic_action[0] == 'move':
                if not _is_move_action_legal(atomic_action, state):
                    return False
            # illegal pick action
            elif atomic_action[0] == 'pick up':
                if not _is_pick_up_action_legal(atomic_action, state):
                    return False
            # illegal drop action
            elif atomic_action[0] == 'drop off':
                if not _is_drop_action_legal(atomic_action, state):
                    return False
            # illegal refuel action
            elif atomic_action[0] == 'refuel':
                if not _is_refuel_action_legal(atomic_action, state):
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

    def apply(self, action, state):
        """
        apply the action to the state
        """
        new_state=deepcopy(state)
        if action == "reset":
            new_state = self.reset_environment(new_state)
            return new_state
        if action == "terminate" or action == None:
            return 'EndOfGame'
        for atomic_action in action:
            new_state = self.apply_atomic_action(atomic_action, new_state)
        return new_state
    def apply_atomic_action(self, atomic_action, state):
        """
        apply an atomic action to the state
        """
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
            state['passengers'][passenger_name]['location'] =state['taxis'][taxi_name]['location']
            state['taxis'][taxi_name]['capacity'] += 1
            return state
        elif atomic_action[0] == 'refuel':
            state['taxis'][taxi_name]['fuel'] = self.initial['taxis'][taxi_name]['fuel']
            return state
        elif atomic_action[0] == 'wait':
            return state
        else:
            raise NotImplemented
    def reset_environment(self, state):
        """
        reset the state of the environment
        """
        state["turns to go"]-=1
        state["taxis"] = self.initial["taxis"]
        state["passengers"] = self.initial["passengers"]
        return state
    def environment_step(self, state):
        """
        update the state of environment randomly
        """
        for p in state['passengers']:
            passenger_stats = state['passengers'][p]
            if np.random.random() < passenger_stats['prob_change_goal']:
                # change destination
                passenger_stats['destination'] = np.random.choice(passenger_stats['possible_goals'])
        state["turns to go"] -= 1
        return

    def build_graph(self):
        """
        build the graph of the problem
        """
        n, m = len(self.initial['map']), len(self.initial['map'][0])
        g = nx.grid_graph((m, n))
        nodes_to_remove = []
        for node in g:
            if self.initial['map'][node[0]][node[1]] == 'I':
                nodes_to_remove.append(node)
        for node in nodes_to_remove:
            g.remove_node(node)
        return g
    def reward(self, action):
        reward = 0
        if action == "reset":
            return -RESET_PENALTY
        if action == "terminate":
            return 0
        for atomic_action in action:
            if atomic_action[0] == 'drop off':
                reward += DROP_IN_DESTINATION_REWARD
            elif atomic_action[0] == 'refuel':
                reward -= REFUEL_PENALTY
        return reward

    def act(self, state):
        return self.best_action(encode(state),self.Q)

    def h1(self,state):
        dist_to_all_passengers={}
        action_for_taxi={}
        for taxi in state['taxis']:
            for passenger in state['passengers']:
                # if you can drop off a passenger, do it
                if state['passengers'][passenger]['location']==taxi and state['taxis'][taxi]['location']==state['passengers'][passenger]['destination']:
                    action_for_taxi[taxi]=('drop off',taxi,passenger)
                    break
                # if you can pick up a passenger, do it
                if state['taxis'][taxi]['location'] == state['passengers'][passenger]['location'] and \
                        state['taxis'][taxi]['capacity'] > 0:
                    action_for_taxi[taxi] = ('pick up', taxi, passenger)
                    break
            # if you can refuel, do it
            if state['taxis'][taxi]['fuel'] < 3:
                action_for_taxi[taxi] = ('refuel', taxi)
                break

