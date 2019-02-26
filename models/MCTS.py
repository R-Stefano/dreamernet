import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS
class Node():
    def __init__(self, state, lstmTuple, parent, parent_action):
        self.state=state
        self.lstmTuple=lstmTuple
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
    def __init__(self, rnn, actor):
        self.num_actions=FLAGS.num_actions
        self.rollouts=FLAGS.rollouts
        self.rnn=rnn
        self.actor=actor

    #This function must return the action to use
    def predict(self, state, lstmTuple):
        #Get the possible actions from the actual state
        priors, value=self.actor.predict(np.concatenate((state, lstmTuple[0,0]), axis=-1))

        #Create the root node
        root_node=Node(state, lstmTuple, None, 0)

        #Generate a list of childs nodes. One for each action
        childs=self.generateChilds(root_node)

        #the priors are the values q(s,a) while the value is np.max(q(s,a))
        #LETS MAKE IT AS THE BEST Q(S,A) in the case of value function actor

        root_node.initialize(priors, value, childs)
        #rollouts to create node's values
        for i in range(self.rollouts):
            nodeToExpand = root_node.selectChild()
            priors, value=self.actor.predict(np.concatenate((nodeToExpand.state, nodeToExpand.lstmTuple[0,0]), axis=-1))
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
        '''
        This function retrieves the state and lstmTuple of the node
        In order to predict all the possible next states from the node state
        '''
        #Retrieve the node state to feed as input
        node_state=parent.state
        #Retrieve the node lstmTuple to initialize the rnn
        node_lstmTuple=parent.lstmTuple

        #Creates a prediction for each action repeating the state
        batchStates=np.repeat(node_state, self.num_actions, axis=0) #num_actions, embed_length

        #Assign an action to each state
        actions=np.asarray([i for i in range(self.num_actions)]).reshape(-1,1)

        #concatenate states and actions to create the input of the rnn
        inputData=np.expand_dims(np.concatenate((batchStates, actions), axis=-1), axis=1)

        #repeate the lstm tuple for each example
        inputInitialize=np.repeat(node_lstmTuple, self.num_actions, axis=-2)

        #The input of the rnn should be [num_actions, 1, enc_state + action] and the lstmTuple
        #Feedx the current node state as well with the possible actions
        #Get the next states based on the actions and create the child nodes
        predictedstates, newLSTMTuple=self.rnn.predict(inputData, initialize=inputInitialize)

        childs=[]        
        for action_idx, s1 in enumerate(predictedstates):
            #format the shape of the rnn predictions (at the moment reward predicted is useless)
            s1=s1[:-1].reshape(1,-1)
            lstmtuple=np.expand_dims(newLSTMTuple[:,:,action_idx], axis=-2)
            #create child nodes for each possible action
            childs.append(Node(s1, lstmtuple, parent, action_idx))
        return childs
