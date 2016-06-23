import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        global action
        action = random.choice(('forward', 'left', 'right'))
        global Q
        Q = [[0.0 for col in range(9999)] for row in range(4)]
 
       
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
       
    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.state = 0    
        dic1 = {None:0, 'forward':1, 'right':2, 'left':3 }
        dic2 = {'green':0, 'red':1}
        self.state = self.state + 1000 * dic1[self.next_waypoint]
        self.state = self.state + 100  * dic2[inputs['light']]
        self.state = self.state + 10   * dic1[inputs['oncoming']]
        self.state = self.state + 1     * dic1[inputs['left']]
        
        # TODO: Select action according to your policy  
        action = random.choice(('forward', 'left', 'right', None))
        epsilon = 0 # in my opinion, epsion is only equal to 0 or 1
        Qold = Q[dic1[action]][int(self.state)]
        
        Qmax = Q[1][int(self.state)]
        actionbest = 'forward'
        if Q[2][int(self.state)] > Qmax:
            Qmax = Q[2][int(self.state)]
            actionbest = 'right'
        if Q[3][int(self.state)] > Qmax:  
            Qmax = Q[3][int(self.state)]
            actionbest = 'left'
        elif Q[0][int(self.state)] > Qmax:
            Qmax = Q[0][int(self.state)]
            actionbest = None
            
        if epsilon:
            action = action
        else: 
            action = actionbest
        
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        alpha = 0.2
        gamma = 0.5
        
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        self.state_1 = 0    
        self.state_1 = self.state_1 + 1000 * dic1[self.next_waypoint]
        self.state_1 = self.state_1 + 100  * dic2[inputs['light']]
        self.state_1 = self.state_1 + 10   * dic1[inputs['oncoming']]
        self.state_1 = self.state_1 + 1     * dic1[inputs['left']]
        
        Qmax = Q[1][int(self.state_1)]
        actionbest = 'forward'
        if Q[2][int(self.state_1)] > Qmax:
            Qmax = Q[2][int(self.state_1)]
            actionbest = 'right'
        if Q[3][int(self.state_1)] > Qmax:  
            Qmax = Q[3][int(self.state_1)]
            actionbest = 'left'
        elif Q[0][int(self.state_1)] > Qmax:
            Qmax = Q[0][int(self.state_1)]
            actionbest = None
  
        Qnew = (1 - alpha) * Qold + alpha * (reward + gamma * Qmax)
        Q[dic1[action]][int(self.state)] = Qnew
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
