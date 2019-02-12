import numpy as np

class Node():
    def __init__(self, state, parent, parent_action):
        self.state=state
        self.priors= []
        self.childs = []
        self.totValue = 0
        self.visits = 0
        self.parent=parent
        self.parent_action=parent_action #stores the action that lead to the node

    #Called when the node is evaluated by the network
    def initialize(self, priors, value, childs):
        self.priors=priors
        self.childs = childs
        self.totValue=value
        self.visits+=1

    def getValue(self,prior):
        if(self.visits > 0):
            q=self.totValue/self.visits
        else:
            q=0
        u=prior/(1+self.visits)
        return q+u

    def selectChild(self):
        #Until a leaf node is not found, search it
        if len(self.childs) !=0:
            #Pick the node with the highest value based on
            # value = totVal/N + p(s,a)/(1 + N)
            values=[]
            for a, node in enumerate(self.childs):
                values.append(node.getValue(self.priors[a]))
            #TODO: HANDLE THE CASE WHERE TWO VALUES ARE EQUAL
            best_action=np.argmax(values)

            #Explore the childs of the node selected
            return self.childs[best_action].selectChild()
        else:
            #Return the node to expand
            return self

    def expand(self, value):
        #if not root_node
        if (self.parent != None):
            self.parent.visits +=1
            self.parent.totValue += value
            self.parent.expand(value)


class Tree():
    def __init__(self, num_actions,rollouts, rnn, predictor):
        self.num_actions=num_actions
        self.rollouts=rollouts
        self.rnn=rnn
        self.predictor=predictor

    #This function must return the action to use
    def predict(self, state):
        priors, value=self.networkSimulator(state)
        root_node=Node(state, None, 0)
        childs=self.generateChilds(root_node)
        root_node.initialize(np.squeeze(priors), np.squeeze(value), childs)
        #rollouts to create node's values
        for i in range(self.rollouts):
            nodeToExpand = root_node.selectChild()
            priors, value=self.networkSimulator(nodeToExpand.state)
            childs=self.generateChilds(nodeToExpand)
            nodeToExpand.initialize(np.squeeze(priors), np.squeeze(value), childs)
            #update the value of the parents
            nodeToExpand.expand(np.squeeze(value))

        #Pick the node with high value and return the action
        actions=[]
        for i, node in enumerate(root_node.childs):
            actions.append(node.getValue(root_node.priors[i]))
        return np.argmax(actions)

    def generateChilds(self, parent):
        #Here retrieve the parents states.
        stateSequence=[]
        dumpParent=parent
        stateSequence.append(np.append(dumpParent.state, -1))
        for t in range(self.rnn.timesteps-1):
            if (dumpParent.parent == None):
                stateSequence.append(np.zeros((self.rnn.state_rep_length + 1)))
            else:
                dumpParent=dumpParent.parent
                stateSequence.append(np.append(dumpParent.state, dumpParent.parent_action))
        
        #numpize the list
        stateSequence=np.expand_dims(np.asarray(stateSequence), axis=0)
        #flip the matrix along timesteps. The last inserted state should be the most recent one
        stateSequence=np.flip(stateSequence, axis=1)

        #Creates a prediction for each action
        batchStates=np.repeat(stateSequence, self.num_actions, axis=0) #num_actions, embed_length

        #Assign an action to each state
        batchStates[:,-1,-1]=np.asarray([i for i in range(self.num_actions)])

        #initialize hidden state and cell state to zeros 
        cell_s=np.zeros((self.num_actions, self.rnn.hidden_units))
        hidden_s=np.zeros((self.num_actions, self.rnn.hidden_units))

        #Feedx the current node state as well with the possible actions
        #Get the next states based on the actions and create the child nodes
        predictedChilds=self.rnn.predict(batchStates, cell_s, hidden_s)
        childs=[]        
        for action_idx, s1 in enumerate(predictedChilds):
            childs.append(Node(s1, parent, action_idx))
        return childs

    def networkSimulator(self, state):
        policy, value=self.predictor.predict(state)
        return policy, value
