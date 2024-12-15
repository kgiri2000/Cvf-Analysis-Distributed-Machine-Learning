import os
import csv
import copy
import math
import itertools
import pandas as pd

class Dijkstra:
    def __init__(self, graph_path, result_path, max_predict):
        self.graph = self.read_graph(graph_path)
        self.graph_name = os.path.splitext(os.path.basename(graph_path))[0]
        self.result_path = result_path
        self.nodes = list(self.graph.keys())
        self.node_positions = {v: i for i, v in enumerate(self.nodes)}
        self.num_of_nodes = len(self.nodes)
        self.nodes_in_num = list(range(self.num_of_nodes))
        self.dataset = []
        self.max_predict = max_predict

        #Only has three colors
        self.colors = list(range(3))
        self.all_configurations = list(itertools.product(self.colors, repeat = self.num_of_nodes))
        self.invariants = set()
        self.program_transitions_rank = {}
        self.program_transitions_n_cvf = {}


        self.pt_rank_effect = {}
        self.cvfs_in_rank_effect = {}
        self.cvfs_out_rank_effect = {}

        self.pt_rank_effect_df = pd.DataFrame()
        self.cvfs_in_rank_effect_df = pd.DataFrame()
        self.cvfs_out_rank_effect_df = pd.DataFrame()

    def read_graph(self, graph_file):
        graph = {}
        with open(graph_file, "r") as f:
            for line in f:
                all_edges = line.split()
                node = all_edges[0]
                edges =  all_edges[1:]
                graph[node] = set(edges)
        return graph

    def is_invariant(self, color):
        bottom = 0
        top = self.num_of_nodes - 1
        eligible_nodes = []

        #Check for bottom
        if (color[bottom] + 1) % 3 == color[bottom + 1]:
            eligible_nodes.append(bottom)
        if (color[top-1]) == color[bottom] and (color[top-1]+1) %3 != color[top]:
            eligible_nodes.append((top-1))
        #Every node between first and last
        for i in range(bottom+1, top):
            if (color[i] +1)%3 == color[i-1]:
                eligible_nodes.append(i)
            if (color[i] +1) % 3 == color[i+1]:
                eligible_nodes.append(i)
        if len(eligible_nodes) != 1:
            return False
        else:
            return True

    def get_invariants(self):
        for state in self.all_configurations:
            if self.is_invariant(state):
                self.program_transitions_rank[state] ={"L": 0,"C": 1, "A": 0, "Ar": 0, "M": 0}
                self.invariants.add(state)
    def is_program_transition(self, position, start_state, perturb_state):
        if start_state in self.invariants and perturb_state in self.invariants:
            return False

        s_start  = start_state[position]
        s_pert = perturb_state[position]

        node = self.nodes[position]

        neighbor_pos = [self.node_positions[n] for n in self.graph[node]]
        neighbor_state = [start_state[i] for i in neighbor_pos]

        left_state, right_state =  neighbor_state

        if node == self.nodes[0]:
            return (s_start+ 1) %  3 == right_state and s_pert == (s_start -1)% 3
        elif node == self.nodes[-1]:
            return (
                left_state == right_state
                and (left_state + 1) % 3 != s_start
                and s_pert == (left_state +1 )% 3
            )
        else:
            if (s_start + 1)% 3 == left_state:
                if s_pert == left_state:
                #if it is true then we return True but if it is false then we don't
                #want to return false as we have to check the another if condition.
                    return True
            if (s_start + 1) % 3 == right_state:
                # we don't have to use if/else because, we don't have to verify any
                #other condition so, we will return either true or false as our final decision
                    return s_pert == right_state
        return False

    def get_program_transitions(self, start_state):
        program_transitions = set()
        for position, _ in enumerate(start_state):
            node_colors = set(range(3))
            for color in node_colors:
                perturb_state = list(start_state)
                perturb_state[position] = color
                perturb_state = tuple(perturb_state)
                if perturb_state != start_state:
                    if self.is_program_transition(position, start_state, perturb_state):
                        program_transitions.add(perturb_state)
        return {"program_transitions": program_transitions}


    def get_cvfs(self, start_state):

        cvfs_in = dict()
        cvfs_out = dict()
        for position, _ in enumerate(start_state):
            for color in self.colors:
                perturb_state = list(start_state)
                perturb_state[position] = color
                perturb_state = tuple(perturb_state)

                if perturb_state != start_state:
                    if start_state in self.invariants:
                        cvfs_in[perturb_state] = position
                    else:
                        cvfs_out[perturb_state] = position

        return{"cvfs_in": cvfs_in, "cvfs_out": cvfs_out}

    def compute_transitions_and_cvfs(self):
        for state in self.all_configurations:
            self.program_transitions_n_cvf[state] = {**self.get_program_transitions(state), **self.get_cvfs(state)}
        return self.program_transitions_n_cvf

    def rank_states(self):
        unranked_states = set(self.program_transitions_n_cvf.keys()) - set(self.program_transitions_rank.keys())
        while unranked_states:
            ranked_states = set(self.program_transitions_rank.keys())
            removed_unranked_state = set()
            for state in unranked_states:
                dests = self.program_transitions_n_cvf[state]['program_transitions']
                if dests - ranked_states:
                    pass
                else:
                    total_path_length = 0
                    path_count = 0
                    max_length = 0
                    for config in dests:
                        path_count += self.program_transitions_rank[config]["C"]
                        total_path_length += self.program_transitions_rank[config]["L"] + self.program_transitions_rank[config]["C"]
                        max_length = max(max_length, self.program_transitions_rank[config]["M"])
                    self.program_transitions_rank[state] = {
                        "L": total_path_length,
                        "C": path_count,
                        "A": total_path_length/path_count,
                        "Ar": math.ceil(total_path_length/path_count),
                        "M": max_length + 1
                    }
                    removed_unranked_state.add(state)
            unranked_states -= removed_unranked_state

    def generate_dataset(self):
        node_count = self.num_of_nodes
        max_pred =  self.max_predict + 2
        #if max is 11, vector size = 11+ 1+1 = 13 -1 = 12(starting from 0)
        for config, values in self.program_transitions_rank.items():
            row = list(config) #3 [0,0,0]
            row += [node_count] #4 [0,0,0,3]
            row += [-1] * (max_pred-len(row)-1) # 13 - 4 -1 = 8 [0,0,0,3,-1,-1 ,-1,-1,-1,-1]
            row += [values["Ar"]] #
            self.dataset.append(row)

    def analyse(self):
        self.get_invariants()
        #print(self.invariants)
        self.compute_transitions_and_cvfs()
        #print(self.program_transitions_n_cvf)
        self.rank_states()
        self.generate_dataset()

        # # Convert the dictionary to a DataFrame
        # df = pd.DataFrame(list(self.program_transitions_n_cvf.items()), columns=['Key', 'Value'])
        # # Define the file path
        # file_path = 'results/out.txt'

        # # Write the DataFrame to a text file with space-separated columns
        # df.to_csv(file_path, sep=' ', index=False)
        #print(self.program_transitions_rank)
        # self.calculate_rank_effect()
        # self.rank_count()
        # self.rank_effect_count()
        # self.rank_effect_individual_nodes()
