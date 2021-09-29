import json
import math
from collections import defaultdict

start = "1"
end = "50"
max_energy = 287932

#***START OF DATA MERGING***

with open('G.json') as f:
    G = json.load(f)

with open('Coord.json') as f:
    Coord = json.load(f)

newdict = {}
x2 = Coord[end][0]
y2 = Coord[end][1]

for i in range(len(G)):
    i+=1
    x1 = Coord[str(i)][0]
    y1 = Coord[str(i)][1]
    h = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))  #Heuristic function
    newdict[str(i)] = h

#Stores the h(n) value of every node into a .json file (e.g. "1": 126981.59164619098, ...)
with open('hvalue.json', 'w') as file_object:
    json.dump(newdict, file_object)

class Graph():
    def __init__(self):
        self.edges = defaultdict(list)
        self.weights = {}
        self.h = {}
        self.costs = {}

    def add_edge(self, from_node, to_node, weight, heuristicfrom, heuristicto, cost):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight
        self.h[from_node] = heuristicfrom
        self.h[to_node] = heuristicto
        self.costs[(from_node, to_node)] = cost
        self.costs[(to_node, from_node)] = cost

graph = Graph()

with open('Dist.json') as f:
    Dist = json.load(f)

with open('Cost.json') as f:
    Cost = json.load(f)

counter = len(Dist)
for i in Dist:
    if counter % 2 == 0:
        fromto = i.split(',')
        graph.add_edge(fromto[0], fromto[1], Dist[i], newdict[fromto[0]], newdict[fromto[1]], Cost[i])
    counter -= 1

#***END OF DATA MERGING***

#***HOW TO USE DATA***
#graph.edges = 'node': ['neighbournode1', 'neighbournode2', ...]
#graph.weights = ('from_node', 'to_node'): edge distance between them
#graph.h = 'node': h(n) or straight line value
#graph.costs = ('from_node', 'to_node'): energy cost between them

#example: graph.edges["1"] = ['2', '12', '1363']
#example: graph.edges["1"][0] = 2
#example: graph.weights["1", "2"] = 803
#example: graph.h["1"] = 126981.59164619098
#example: graph.costs["1", "2"] = 2008

#***START OF A-STAR ALGORITHM***
def astar():
    #Initialization
    openlist = set() #openlist stores visited nodes that have *unvisited* neighbour nodes
    openlist.add(start) #put start node on openlist
    closelist = set() #closelist stores visited nodes that neighbour nodes *are all visited*

    gvalue = {} #gvalue stores the accumulated g(n) value of the node
    gvalue[start] = 0 #set the g(n) of start node to 0

    energy = {} #energy stores the accumulated energy cost of the node
    energy[start] = 0 #set the energy cost of start node to 0

    prev = {} #prev stores the previous node of the node
    prev[start] = start #set the previous node of the start node to itself

    while len(openlist) > 0:
        currentnode = ""

        #find node with lowest value of f(n) in the openlist
        for node in openlist:
            # if there is no current node (when starting) or there is a node with lower f(n) in the openList, set the current node as node
            if currentnode == "" or gvalue[node] + graph.h[node] < gvalue[currentnode] + graph.h[currentnode]: #f(n) = g(n) + h(n)
                currentnode = node

        #if there is nothing in the openList, path is invalid
        if currentnode == "":
            print("Invalid path")
            return

        #TERMINATING CONDITION: end node reached
        if currentnode == end:
            shortestdist = gvalue[currentnode] #Shortest distance
            totalenergy = energy[currentnode] #Total energy

            path = []

            while prev[currentnode] != currentnode:
                path.append(currentnode)
                currentnode = prev[currentnode]

            path.append(start)
            path.reverse()

            print("Shortest path: ", end="")
            for i in path:
                if i != end:
                    print(i, end="->")
                else:
                    print(i)
            print("Shortest distance: ", shortestdist)
            print("Total energy cost: ", totalenergy)
            return

        #loop through each neighbour node of the current node
        for neighbournode in graph.edges[currentnode]:
            #if the neighbour node is not found in the openlist or closelist
            if neighbournode not in openlist and neighbournode not in closelist:
                #Calculate the energy cost needed to get to neighbour node
                #energy of neighbour node = accumulated energy of current node + energy cost between neighbour node and current node
                energy[neighbournode] = energy[currentnode] + graph.costs[neighbournode, currentnode]
                #Check if energy cost needed to get to neighbour node > max_energy (energy budget), if > max_energy, skip this neighbour node and move to next neighbour node
                if energy[neighbournode] > max_energy:
                    continue

                openlist.add(neighbournode) #add neighbour node to openlist
                prev[neighbournode] = currentnode #set neighbour node's previous node as current node

                #Calculate the g(n) needed to get to neighbour node
                #g(n) of neighbour node = accumulated g(n) of current node + g(n) between neighbour node and current node
                gvalue[neighbournode] = gvalue[currentnode] + graph.weights[neighbournode,currentnode]

            #if neighbour node is in the openlist or closelist
            else:
                #Check if g(n) of neighbour node (OLD VALUE) > accumulated g(n) of current node + g(n) between neighbour node and current node (NEW VALUE)
                if gvalue[neighbournode] > gvalue[currentnode] + graph.weights[neighbournode,currentnode]:
                    #Since the OLD g(n) is > the NEW g(n), we update with the NEW g(n) (SHORTER DISTANCE)
                    gvalue[neighbournode] = gvalue[currentnode] + graph.weights[neighbournode,currentnode]
                    # Calculate the energy cost needed to get to neighbour node
                    # energy of neighbour node = accumulated energy of current node + energy cost between neighbour node and current node
                    energy[neighbournode] = energy[currentnode] + graph.costs[neighbournode, currentnode]
                    #Check if energy cost needed to get to neighbour node > max_energy (energy budget), if > max_energy, skip this neighbour node and move to next neighbour node
                    if energy[neighbournode] > max_energy:
                        continue

                    prev[neighbournode] = currentnode #set neighbour node's previous node as current node

                    #Check if neighbour node is in the closelist, if yes, move it to openlist
                    if neighbournode in closelist:
                        closelist.remove(neighbournode)
                        openlist.add(neighbournode)

        #remove current node from openlist and add to closelist
        openlist.remove(currentnode)
        closelist.add(currentnode)

    print("Invalid path")
    return

#***END OF A-STAR ALGORITHM***

print("***A-STAR ALGORITHM***")
astar()
