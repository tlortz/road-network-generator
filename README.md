A simple Python module that can be used to create a network of points resembling an urban street network. 

Sample usage:
```
from generate_roads import *
network1 = Network(20,[0,0,1],[1,1,.1])
network1.evolve()
network1.get_incumbent()
```
