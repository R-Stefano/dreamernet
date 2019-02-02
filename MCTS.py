import numpy as np
import random
class Node():
    def __init__(self, state, num_actions, reward):
        self.state=state
        self.avail_actions=[i for i in range(num_actions)]
        self.childs = {}
        self.value = reward
        self.expansions = 0
        self.num_actions=num_actions

        self.explorationRate=0.2

    def expand(self):
        #Randomly pick one of the available actions
        action=random.choice(self.avail_actions)

        #SIMULATOR
        new_state=np.random.randint(0,255,(64,64))
        reward=np.random.random() -0.5
        self.addChild(Node(new_state, self.num_actions, reward), action)

    def addChild(self, node, action):
        print('expanding node. Create node', node)
        self.childs[action]=node
        #Remove avail action in the node
        del self.avail_actions[self.avail_actions.index(action)]

    def selectChild(self):
        #HERE DECIDE IF SELECT NEW CHILD OR EXPAND EXISTING ONE
        if len(self.childs) !=0:
            #don't wait leaf node, randomly expand current node
            #with available actions
            if(np.random.random()<self.explorationRate and len(self.avail_actions)>0):
                print('randomly expanding node (not leaf)', self)
                self.expand()
            else:
                #Select the node with highest value
                print('selecting among childs of', self)
                node=list(self.childs.items())[0][1]
                node.selectChild()
        else:
            print('leaf node found', self)
            self.expand()

    def updateValue(self, reward):
        self.expansions +=1
        self.value += reward

    def describ(self):
        print('Childs',self.childs)
        print('Value',self.value)
        print('Iterations',self.expansions)

class Tree():
    def __init__(self, num_actions):
        self.num_actions=num_actions
    #This function must return the action to use
    '''
    Once instantiated the root node,
    I need to expand it creating the childs.

    How create the childs?
    INstantianting other nodes.

    These nodes must be associated with the action
    that created them. So, before creating the child,
    I have to select the action

    How can I select the action?
    Starting from the node to expand, the action
    is selected randomlyself.

    So, I select randomly an action in order to
    expand the tree.

    NOw, with the action, i fed into the simulator ù
    retrieving the state and reward.

    Now, I have the info, for creating another nodeself.

    Next, the node is associated with the parent node, using
    the action that created it. Then, I repeat the process.

    '''
    def predict(self, state):
        root_node=Node(state, self.num_actions, 0)

        #make 10 rollouts to create node's value
        for i in range(10):
            print('Rollout', i+1)
            root_node.selectChild()
            print('\n')

        #Pick the node with high value and return the action
