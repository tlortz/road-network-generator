import numpy as np
from scipy.sparse import *
from scipy.spatial.distance import euclidean
import uuid

class Point:
    def __init__(self,coords):
        self.coords = coords
        
class Network:
    def __init__(self,num_points,means,stdevs):
        self.num_points = num_points
        self.points = [Point(np.random.normal(means,stdevs)) for i in range(num_points)]
        self.all_pairs = self.make_all_pairs()
        self.all_distances = self.get_all_distances([p.coords for p in self.points],self.all_pairs)
        self.csr_matrix_full = self.make_csr_matrix(self.all_pairs,self.all_distances,[1 for i in range(len(self.all_pairs))])
        self.min_span_tree = csgraph.minimum_spanning_tree(self.csr_matrix_full).sign()#.data
        self.min_network = []
        for i in range(self.min_span_tree.shape[0]-1):
            for j in range(i+1,self.min_span_tree.shape[1]):
                self.min_network.append(int(self.min_span_tree.getrow(i).getcol(j).todense()))
        self.max_distance = np.sum(self.all_distances)
        self.max_network = [1 for p in self.all_pairs]
        self.candidate_dict = {} #give each candidate:
        max_candidate = self.generate_candidate_from_connections(self.max_network)
        min_candidate = self.generate_candidate_from_connections(self.min_network)
        self.candidate_dict[max_candidate['uuid']] = max_candidate
        self.candidate_dict[min_candidate['uuid']] = min_candidate
        if max_candidate['fitness'] < min_candidate['fitness']:
            self.top_candidate = max_candidate
            self.bottom_candidate = min_candidate
        else: 
            self.top_candidate = min_candidate
            self.bottom_candidate = max_candidate
        
    def make_all_pairs(self):
        #returns a list of pairs intended to be treated as an undirected graph
        pairs = []
        for p in range(self.num_points-1):
            for q in range((p+1),self.num_points):
                pairs.append((p,q))
        return pairs
    
    def get_all_distances(self,points,edges,connections=None):
        if connections==None:
            connections = np.ones(len(edges))
        valid_edges = [edges[i] for i in range(len(edges)) if connections[i]==1]
        distances = [euclidean(points[e[0]],points[e[1]]) for e in valid_edges]
        return distances
    
    def make_csr_matrix(self,pairs,distances,connections):
        length = self.num_points
        distances_updated = [list(distances)[i] for i in range(len(distances)) if list(connections)[i]>0]
        pairs_updated = [list(pairs)[i] for i in range(len(pairs)) if list(connections)[i]>0]
        row = np.append([p[0] for p in pairs_updated],[i for i in range(length)])
        col = np.append([p[1] for p in pairs_updated],[i for i in range(length)])
        data = np.append(distances_updated,[0 for i in range(length)])
        matrix = csr_matrix((data,(row,col)))
        return matrix
    
    def generate_candidate_from_connections(self,connections):
        uuid_ = str(uuid.uuid4())
        csr_matrix = self.make_csr_matrix(self.all_pairs,self.all_distances,connections)
        csr_matrix_binary = csr_matrix.sign()
        num_components = csgraph.connected_components(csr_matrix_binary)[0]
        total_distance = np.sum(csr_matrix)
        connection_counts = []
        for i in range(csr_matrix_binary.shape[0]):
            num_connections = np.sum(csr_matrix_binary.getrow(i)) + np.sum(csr_matrix_binary.getcol(i))
            connection_counts.append(num_connections)
        candidate = {}
        candidate['uuid'] = uuid_
        candidate['connections'] = connections
        candidate['csr_matrix'] = csr_matrix
        candidate['num_components'] = num_components
        candidate['total_distance'] = total_distance
        candidate['connection_counts'] = connection_counts
        candidate['fitness'] = self.evaluate_fitness(candidate)
        return candidate
        
    def fitness_connectivity(self,candidate):
        return candidate.get('num_components')
    
    def fitness_distance(self,candidate):
        return candidate.get('total_distance')/self.max_distance
    
    def fitness_intersections(self,candidate):
#         return np.mean(np.abs(np.reshape(candidate.get('connection_counts'),-1)-4.0))
        return np.linalg.norm(np.reshape(candidate.get('connection_counts'),-1)-4.0)
        
    def evaluate_fitness(self,candidate):
        return 1/self.fitness_connectivity(candidate) + 2/self.fitness_distance(candidate) + 4/self.fitness_intersections(candidate)
    
    def select_parents(self):
        ids = [self.candidate_dict.get(c).get('uuid') for c in self.candidate_dict.keys()]
        scores = [self.candidate_dict.get(c).get('fitness') for c in self.candidate_dict.keys()]
        total_score = np.sum(scores)
        probs = scores/total_score
        parent_ids = np.random.choice(ids,size=2,replace=False,p=probs)
        return parent_ids
    
    def create_offspring(self,parent_ids):
        # from two candidate parents, generate each feature based on a random selection
        # from the two parents, weighted by the parents' fitness scores
        offspring_connections = self.candidate_dict.get(parent_ids[0]).get('connections')
        p0 = self.candidate_dict.get(parent_ids[0]).get('fitness')
        p1 = self.candidate_dict.get(parent_ids[1]).get('fitness')
        p_tot = p0+p1
        p_change = p1/p_tot
        for i in range(len(offspring_connections)):
            if np.random.uniform() < p_change:
                offspring_connections[i] = self.candidate_dict.get(parent_ids[1]).get('connections')[i]
        return offspring_connections
    
    def mutate(self,connections,p_mutate=.1):
        # for each of the elements in the candidate, change their value 
        # with probability mutate_percent
        valid = False
        while not valid:
            for i in range(len(connections)):
                if np.random.uniform() < p_mutate:
                    if connections[i]==1:
                        connections[i]=0
                    else:
                        connections[i]=1
            if np.sum(connections) > 0:
                valid = True
        return connections
    
    def evolve(self,pop_size=2000,no_change_threshold=100,max_iterations=50000):
        num_unchanged = 0
        num_iterations = 0
        while (num_unchanged < no_change_threshold and num_iterations < max_iterations):
            new_connections = self.mutate(self.create_offspring(self.select_parents()))
            new_candidate = self.generate_candidate_from_connections(new_connections)
            if len(self.candidate_dict) < pop_size:
                self.candidate_dict[new_candidate['uuid']] = new_candidate
                if new_candidate['fitness']<self.bottom_candidate['fitness']:
                    self.bottom_candidate = new_candidate
            else:
                if new_candidate['fitness']>self.bottom_candidate['fitness']:
                    # drop current bottom
                    self.candidate_dict.pop(self.bottom_candidate['uuid'])
                    # add new_candidate
                    self.candidate_dict[new_candidate['uuid']] = new_candidate
                    # find new bottom
                    ids = list(self.candidate_dict.keys())
                    scores = [self.candidate_dict.get(k).get('fitness') for k in ids]
                    min_ix = np.argmin(scores)
                    min_id = ids[min_ix]
                    self.bottom_candidate = self.candidate_dict.get(min_id)
                    num_unchanged = 0
                else:
                    # don't add new candidate, increase
                    num_unchanged = num_unchanged + 1
            if new_candidate['fitness']>self.top_candidate['fitness']:
                    self.top_candidate = new_candidate
            num_iterations = num_iterations + 1
    
    def get_incumbent(self):
        self.print_candidate(self.top_candidate.get('uuid'))
    
    def print_candidate(self,uuid):
        result = ""
        cd = self.candidate_dict[uuid]
        points = [p.coords for p in self.points]
        for point in points:
            result = result + str(point[0]) + "," + str(point[1]) + "," + str(point[2]) + "\n"
        connections = [self.all_pairs[i] for i in range(len(self.all_pairs)) if cd['connections'][i]>0]
        for edge in connections:
            result = result + str(edge[0]) + "," + str(edge[1]) + "\n"
        print(result)
    
    def reset_network(self):
        # remove the entire set of candidates, incumbent and bottom scores
        # rebuild the candidate_dict, incumbent and bottom from the two original candidates
        pass