import numpy as np


class Node():
    def __init__(self, state, parent, action, h=0, generation=0):
        self.state = state
        self.parent = parent
        self.action = action
        
        # Heurestics
        
        # Manhattan distance
        self.h = h
        
        # Steps of execution
        self.generation = generation
        
        # Heuristic cost
        self.cost = self.h + self.generation

class StackFrontier():
    def __init__(self):
        self.frontier = []
        
    def add(self, node):
        self.frontier.append(node)
        
    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)
    
    def empty(self):
        return len(self.frontier) == 0
    
    def remove(self):
        if self.empty():
            raise Exception("empty frontier")
        else:
            node = self.frontier[-1]
            self.frontier = self.frontier[:-1]
            return node
        
    def select(self):
        """Selecting the node from the frontier based on minimum cost"""
        node_cost = dict()
        if self.empty():
            raise Exception("empty frontier")
        else:
            for node in self.frontier:
                node_cost[node.cost] = node
                
        minimum_cost = min(node_cost.keys())
        
        self.frontier.remove(node_cost[minimum_cost])
        
        return node_cost[minimum_cost]
    
class Astar():
    def __init__(self, maze, start, end):
        self.maze = np.array(maze)
        self.start = start
        self.end = end
        
        self.height = self.maze.shape[0]
        self.width = self.maze.shape[1]
        
        # Keep track of walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if maze[i][j] == 0:
                        row.append(True)
                    else:
                        row.append(False)
                except IndexError:
                    row.append(True)
            self.walls.append(row)
            
        self.solution = None
        
    def neighbors(self, state):
        row, col = state
        
        candidates = [
            ("n", (row - 1, col)),
            ("s", (row + 1, col)),
            ("w", (row, col - 1 )),
            ("e", (row, col + 1)),
            ("ne", (row - 1, col + 1)),
            ("se", (row + 1, col + 1)),
            ("nw", (row - 1, col - 1)),
            ("sw", (row + 1, col - 1))
        ]
        
        result = []
        
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r,c)))
        return result
    
    def solve(self, verbose=True):
        """Finds a solution to maze, if one exists."""
        
        # Keep track of number of states explored
        self.num_explored = 0
        
        # Initialize frontier to just the starting state
        start = Node(state=self.start, parent=None, action=None, generation=0)
        frontier = StackFrontier()
        frontier.add(start)
        
        # Initialize an empty explored set
        self.explored = set()
        
        # Keep looping until solution is found
        while True:
            # If nothing left in frontier, then no path
            if frontier.empty():
                raise Exception("no solution")
                
            # Choose a node from the frontier
            node = frontier.select()
            self.num_explored += 1
        
            if verbose:
                print(f"Node state: {node.state}, manhattan distance: {node.h}, generation: {node.generation}, cost: {node.cost}")
            
            # If node is the goal, then we have a solution
            if node.state == self.end:
                actions = []
                cells = []
                
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.state)
                    node = node.parent
                
                actions.reverse()
                cells.reverse()
                self.solution = (actions, cells)
                return
            
            # Mark node as explored
            self.explored.add(node.state) 
            
            # Add neighbors to frontier
            for action, state in self.neighbors(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    # calculating manhattan distance h()
                    manhattan_distance = np.abs(state[0] - self.end[0]) + np.abs(state[1] - self.end[1])
                    
                    child = Node(state=state, 
                                 parent=node,
                                 action=action,
                                 h=manhattan_distance, 
                                 generation=node.generation+1)
                                
                    frontier.add(child)