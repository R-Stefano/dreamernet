import numpy as np

class Node():
    def __init__(self, state, parent):
        self.state=state
        self.priors= []
        self.childs = []
        self.totValue = 0
        self.visits = 0
        self.parent=parent

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
    def __init__(self, num_actions,rollouts, rnn):
        self.num_actions=num_actions
        self.rollouts=rollouts
        self.rnn=rnn

    #This function must return the action to use
    def predict(self, state):
        priors, value=self.networkSimulator(state)
        root_node=Node(state, None)
        childs=self.generateChilds(root_node)
        root_node.initialize(priors, value, childs)

        #rollouts to create node's values
        for i in range(self.rollouts):
            nodeToExpand = root_node.selectChild()
            priors, value=self.networkSimulator(nodeToExpand.state)
            childs=self.generateChilds(nodeToExpand)
            nodeToExpand.initialize(priors, value, childs)
            #update the value of the parents
            nodeToExpand.expand(value)

        #Pick the node with high value and return the action
        actions=[]
        for i, node in enumerate(root_node.childs):
            actions.append(node.getValue(root_node.priors[i]))
        return np.argmax(actions)

    def generateChilds(self, parent):
        #Here retrieve the parents states.
        stateSequence=[]
        dumpParent=parent
        stateSequence.append(dumpParent.state)
        for t in range(self.rnn.timesteps-1):
            if (dumpParent.parent == None):
                print('Check None parent', dumpParent.parent)
                stateSequence.append(np.zeros((rnn.state_rep_length)))
            else:
                dumpParent=dumpParent.parent
                stateSequence.append(dumpParent.state)

        print(stateSequence)

        #Creates a prediction for each action
        batchStates=np.repeat(np.expand_dims(parent.state, axis=0), self.num_actions, axis=0) #num_actions, embed_length

        #Assign an action to each state
        #TODO: CONVERT TO NUMPY
        batchActions=np.expand_dims(np.asarray([i for i in range(self.num_actions)]), axis=1)

        inputData=np.concatenate((batchStates, batchActions), axis=-1)

        #initialize hidden state and cell state to zeros 
        cell_s=np.zeros((test_batch_size, rnn.hidden_units))
        hidden_s=np.zeros((test_batch_size, rnn.hidden_units))

        #TODO: RETRIEVE THE PARENT STATES. I NEED A SEQUENCE OF 10 STATES OR PADDING
        predictedChilds=self.rnn.predict(inputData, cell_s, hidden_s)
        print(predictedChilds.shape)
        childs=[]        
        for s1 in predictedChilds:
            print(s1)
            childs.append(Node(s1, parent))
        return childs

    def networkSimulator(self, state):
        output=np.random.randint(0,10,(self.num_actions))
        return output/np.sum(output), np.random.random()*2 -1.0
