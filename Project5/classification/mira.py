# mira.py
# -------
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


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        bestAccuracy = None
        bestWeights = None
        bestC = None

        startingWeights = self.weights

        for C in Cgrid:
            self.weights = startingWeights
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):
                    labelScores = [self.weights[label] * trainingData[i]
                        for label in self.legalLabels]
                    predictedLabelIndex = labelScores.index(max(labelScores))
                    predictedLabel = self.legalLabels[predictedLabelIndex]

                    if predictedLabel != trainingLabels[i]:
                        w = self.weights[predictedLabel]
                        wStar = self.weights[trainingLabels[i]]
                        f = trainingData[i]

                        tau = ((w - wStar) * f + 1.0) / (2 * (f * f))
                        tau = min(C, tau)

                        for j in f:
                            w[j] -= tau * f[j]
                            wStar[j] += tau * f[j]

            # Evaluate accuracy for the current value of C
            correctPredictions = 0
            for i in range(len(validationData)):
                labelScores = [self.weights[label] * validationData[i]
                    for label in self.legalLabels]
                predictedLabelIndex = labelScores.index(max(labelScores))
                predictedLabel = self.legalLabels[predictedLabelIndex]

                if predictedLabel == validationLabels[i]:
                    correctPredictions += 1

            accuracy = float(correctPredictions) / len(validationData)

            if bestAccuracy is None or accuracy >= bestAccuracy:
                # Favor lower values of C in tie-breaks
                if accuracy == bestAccuracy and C > bestC:
                    continue

                bestC = C
                bestAccuracy = accuracy
                bestWeights = self.weights

        self.weights = bestWeights

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses
