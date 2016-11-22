# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newScore = successorGameState.getScore()
        remainingFood = len([food for food in newFood if food is True])
        evaluation = newScore + 100 / (1 + remainingFood)

        closestFoodDistance = 99999
        for food in currentGameState.getFood().asList():
            foodDistance = (((food[0] - newPos[0])**2 +
                (food[1] - newPos[1])**2) ** 0.5)
            if foodDistance < closestFoodDistance:
                closestFoodDistance = foodDistance

        evaluation += 100 / (1 + closestFoodDistance)

        for ghostState in newGhostStates:
            ghostConfiguration = ghostState.configuration
            ghostDistance = manhattanDistance(ghostConfiguration.pos, newPos)
            evaluation -= 100 / (1 + ghostDistance**3)

        return evaluation

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.inf = 99999
        bestAction, bestValue = self.minimax(gameState, 0, self.depth)
        # print "minimax depth {}, value {}".format(self.depth, bestValue)

        return bestAction

    def minimax(self, gameState, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return (None, self.evaluationFunction(gameState))

        nextAgent = agent + 1
        if nextAgent >= gameState.getNumAgents():
            # roll back to pacman and reduce depth
            nextAgent = 0
            depth -= 1

        bestAction = None
        bestValue = None
        for action in gameState.getLegalActions(agent):
            actionValue = self.minimax(
                gameState.generateSuccessor(agent, action), nextAgent, depth)[1]

            if bestValue is None:
                bestAction = action
                bestValue = actionValue
            else:
                if ((agent == 0 and actionValue > bestValue)
                or (agent != 0 and actionValue < bestValue)):
                    bestAction = action
                    bestValue = actionValue

        return (bestAction, bestValue)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        self.inf = 99999
        bestAction, bestValue = self.alphabeta(gameState, 0, self.depth)
        # print "alphabeta depth {}, value {}".format(self.depth, bestValue)

        return bestAction

    def alphabeta(self, gameState, alpha, beta, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return (None, self.evaluationFunction(gameState))

        if agent == 0:
            return self.max_value(gameState, alpha, beta, agent, depth)
        else:
            return self.min_value(gameState, alpha, beta, agent, depth)

    def max_value(self, gameState, alpha, beta, agent, depth):
        nextAgent = agent + 1

        bestValue = -self.inf
        for action in gameState.getLegalActions(agent):
            actionValue = self.minimax(
                gameState.generateSuccessor(agent, action),
                alpha, beta, nextAgent, depth)[1]

            if actionValue > bestValue:
                if actionValue > beta:
                    return (action, actionValue)
                bestAction = action
                bestValue = actionValue
            alpha = max(alpha, bestValue)

        return (bestAction, bestValue)

    def min_value(self, gameState, alpha, beta, agent, depth):
        nextAgent = agent + 1
        if nextAgent >= gameState.getNumAgents():
            # roll back to pacman and reduce depth
            nextAgent = 0
            depth -= 1

        bestValue = self.inf
        for action in gameState.getLegalActions(agent):
            actionValue = self.minimax(
                gameState.generateSuccessor(agent, action),
                alpha, beta, nextAgent, depth)[1]

            if actionValue < bestValue:
                if actionValue < alpha:
                    return (action, actionValue)
                bestAction = action
                bestValue = actionValue
            beta = min(beta, bestValue)

        return (bestAction, bestValue)

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        self.inf = 99999
        bestAction, bestValue = self.expectimax(gameState, 0, self.depth)
        print "expectimax depth {}, value {}".format(self.depth, bestValue)

        return bestAction

    def expectimax(self, gameState, agent, depth):
        if gameState.isWin() or gameState.isLose() or depth <= 0:
            return (None, self.evaluationFunction(gameState))

        nextAgent = agent + 1
        if nextAgent >= gameState.getNumAgents():
            # roll back to pacman and reduce depth
            nextAgent = 0
            depth -= 1

        bestValue = -self.inf
        valueSum = 0.0
        valueCount = 0
        for action in gameState.getLegalActions(agent):
            actionValue = self.expectimax(
                gameState.generateSuccessor(agent, action), nextAgent, depth)[1]

            if actionValue > bestValue:
                bestAction = action
                bestValue = actionValue
            valueSum = valueSum + actionValue
            valueCount += 1

        if agent == 0:
            return (bestAction, bestValue)
        else:
            return (None, valueSum/valueCount)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
