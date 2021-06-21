import itertools
import copy
import numpy as np
from collections import OrderedDict


class Agent(object):
    def __init__(self, cm, beliefs, n, R, S, lexicon):
        self.cm = cm
        self.beliefs = beliefs
        self.n = n
        self.R = R
        self.S = S
        self.lexicon = lexicon

    def delta_speaker(self, s, r):
        sum = 0.0
        for sig in self.S:
            sum += self.lexicon[sig][r]
        output = self.lexicon[s][r] / sum
        return output

    def delta_listener(self, s, r):
        sum = 0.0
        for ref in self.R:
            sum += self.lexicon[s][ref]
        output = self.lexicon[s][r] / sum
        return output

    def pr_ll(self, s, r, n):
        if n == 0:
            return self.delta_listener(s, r)
        else:
            sum = 0.0
            for ref in self.R:
                sum += self.pr_ls(s, ref, n)
            return self.pr_ls(s, r, n) / sum

    def pr_ls(self, s, r, n):
        sum = 0.0
        for sig in self.S:
            sum += self.pr_ll(sig, r, n - 1)
        return self.pr_ll(s, r, n - 1) / sum

    def pr_ss(self, s, r, n):
        if n == 0:
            return self.delta_speaker(s, r)
        else:
            sum = 0.0
            for sig in self.S:
                sum += self.pr_sl(sig, r, n)
            return self.pr_sl(s, r, n) / sum

    def pr_sl(self, s, r, n):
        sum = 0.0
        for ref in self.R:
            sum += self.pr_ss(s, ref, n - 1)
        return self.pr_ss(s, r, n - 1) / sum

    def coherence_value(self, cm, beliefs):
        """
        Calculates the coherence value of a coherence graph given the connections in the matrix and the truth values.
        Satisfied constraints contribute +1, unsatisfied constrains 0.
        :param cm: The weights of the network: a 2D array of the connections between the nodes, 1 is a positive constraint
        between nodes, -1 a negative,
        and 0 is when there is no connection. Furthermore, nodes are not connected with themselves.
        :param beliefs: List of booleans, each boolean is the value of a node, can be None
        :return: The coherence value of the given network
        """
        cm_copy = copy.deepcopy(cm)  # Make a copy of the coherence matrix so we don't modify the original one by
        # reference.
        coh = 0
        for i in range(len(beliefs)):
            for j in range(len(beliefs)):
                if not i == j:  # Nodes are not connected with themselves.
                    if not beliefs[i] is None and not beliefs[j] is None:
                        if beliefs[i] == beliefs[j] and cm_copy[i][j] == 1:  # Same belief & positive constraint
                            coh += 1
                        if not beliefs[i] == beliefs[j] and cm_copy[i][j] == -1:  # Different belief, negative
                            # constraint.
                            coh += 1
                    cm_copy[j][i] = 0  # cm_copy[i][j] = cm_copy[j][i], so weight is set to 0 to avoid counting same
                    # edge twice.
        return coh

    def random_none_override(self, beliefs):
        """
        This function overrides any None value in a beliefs set with randomly chosen values of either True or False
        :param beliefs:
        :return:
        """
        np.random.seed()
        possibilities = [True, False]
        for index, belief in enumerate(beliefs):
            if belief is None:
                beliefs[index] = np.random.choice(possibilities)
        return beliefs


    def optimal_coherence_network(self, cm, beliefs):
        """
        Given a coherence network with the weights and a set of beliefs with possible null / None entries, this function
        finds the optimal values for the null entries to create the highest possible coherence for the network.
        :param cm: Coherence matrix
        :param beliefs: Beliefs (True or False or None) for each node
        :return: The optimal set of truth values for this coherence matrix
        """
        # First every possible combination of {True, False} is found for the nodes of the network
        every_possibility = [list(i) for i in itertools.product([True, False], repeat=len(beliefs))]
        # Then the values of the nodes for which a belief exists (not None) is overwritten with the belief available
        for pos in every_possibility:
            for i in range(len(beliefs)):
                if beliefs[i] == True:
                    pos[i] = True
                if beliefs[i] == False:
                    pos[i] = False
        # Now duplicates are removed from the resulting list, and we are left with a list which has the beliefs from
        # the input, but for the values where the was a None, either True or False is inserted to account for every
        # resulting option.
        corrected_possibilities = [i for n, i in enumerate(every_possibility) if i not in every_possibility[:n]]
        print(corrected_possibilities)

        coh = -1
        optimal = []
        for option in corrected_possibilities:
            coh_new = self.coherence_value(cm, option)
            if coh_new > coh:
                coh = coh_new
                optimal = option
        return optimal

    def speak(self, ref):
        """
        Returns for a chosen referent the signal which is most likely to be correctly interpreted based on n-order RSA
        It returns this is a tuple along with the opposite of the belief which the speaker has associated with this referent
        :param ref: Chosen referent
        :return: Tuple : (signal, boolean)
        """
        np.random.seed()
        possibilities = [True, False]
        if self.beliefs[ref] is None: # If the agent has no belief on this referent, the lie will be randomly decided
            bool = np.random.choice(possibilities)
        else:
            bool = not (self.beliefs[ref])  # The boolean is the opposite of the speaking agent's belief
        final_signal = None
        prob = -1.0
        probs_and_signals = [[self.pr_ss(s, ref, self.n), s] for s in
                             self.S]  # All possible signals and their probabilities under n-RSA
        for probsig in probs_and_signals:  # This loop selects the signal with the highest probability
            if probsig[0] > prob:
                final_signal = probsig[1]
                prob = probsig[0]
        return final_signal, bool

    def listen(self, s_obs):
        """
        Returns the the referent that is most likely intended based on n-order RSA and returns the boolean from the input
        along with it.
        :param s_obs: a tuple of (signal, boolean)
        :return: a tuple of (referent, boolean)
        """
        signal, bool = s_obs
        final_referent = None
        prob = -1.0
        probs_and_refs = [[self.pr_ll(signal, r, self.n), r] for r in self.R]
        for probref in probs_and_refs:
            if probref[0] > prob:
                final_referent = probref[1]
                prob = probref[0]
        return final_referent, bool

    def lie_detection(self, message):
        """
        For the given input a tuple of an percieved intended referent and a boolean (belief) associated with that referent,
        the coherence value is calculated for the current set of beliefs of the listening agent, and then again with the
        updated beliefs from the input. If the coherence value is higher for the current set of beliefs, then the message
        is percieved as a lie, and true is returned, else false.
        :param message: a tuple of (referent, boolean)
        :return: a boolean
        """
        fill_in_nones = True  # If this is True, the None beliefs are filled in randomly to be True or False
        referent, bool = message
        if not fill_in_nones:
            updated_beliefs_l = copy.deepcopy(self.beliefs)
            updated_beliefs_l[referent] = bool
            coh_now = self.coherence_value(self.cm, self.beliefs)
            coh_updated = self.coherence_value(self.cm, updated_beliefs_l)
            lie_detected = coh_now > coh_updated
        else:
            beliefs_copy = copy.deepcopy(self.beliefs)
            filled_in_beliefs = self.random_none_override(beliefs_copy)
            updated_beliefs_l = copy.deepcopy(filled_in_beliefs)
            updated_beliefs_l[referent] = bool
            coh_now = self.coherence_value(self.cm, filled_in_beliefs)
            coh_updated = self.coherence_value(self.cm, updated_beliefs_l)
            lie_detected = coh_now > coh_updated
        return lie_detected


class SpeakerAgent(Agent):
    def __init__(self, cm, beliefs, n, R, S, lexicon):
        super().__init__(cm, beliefs, n, R, S, lexicon)


class ListenerAgent(Agent):
    def __init__(self, cm, beliefs, n, R, S, lexicon):
        super().__init__(cm, beliefs, n, R, S, lexicon)
