# coding=utf-8
import itertools
from collections import OrderedDict
import agent
import numpy as np
import math
import copy
import random

"""
README:

In the main_program() there is the code which activates the main data gathering code (new_gather_data).
Some things are commented out as well, which are tests which were run for testing purposes.

If you want to run the simulations with replacing the None values, then this can be done in lie_detection function
in the agent.py class.
There you can set the boolean fill_in_nones to either True or False, depending on what you want to run.
"""

def new_gather_data(n_beliefs, n_order):
    """
    New function to gather data. Makes exhaustive beliefs data set whereas the old one did not.
    Makes a fixed lexicon, 20 random coherence network and exhaustive belief data sets as specified in input.
    Runs simulations for these settings, n order as specified in input.
    Records when the lie is detected and the total number of trials.
    Prints the results (amount of lie detected, total trials per beliefs set)
    :param n_beliefs:
    :param n_order:
    :return:
    """
    lexicon_local = [[1, 0, 0, 1, 1, 0, 1, 0],
               [0, 1, 0, 1, 0, 0, 1, 1],
               [1, 1, 0, 0, 0, 1, 1, 1],
               [1, 1, 0, 1, 1, 0, 1, 0],
               [0, 1, 0, 0, 0, 1, 0, 1],
               [1, 0, 1, 1, 0, 0, 0, 0],
               [0, 0, 0, 1, 1, 0, 1, 1],
               [1, 1, 0, 1, 0, 0, 1, 0]]

    R_local = [i for i in range(n_beliefs)]
    S_local = [i for i in range(n_beliefs)]
    np.random.seed()

    print("Making CMs...")
    # Make CMs
    cm_list = []
    for i in range(20):
        cm_list.append(make_random_cm(n_beliefs))
    print("Done making CMs!")

    # Make every possible beliefs set for 0 to n_beliefs - 1 nones per beliefs set
    print("Making belief sets...")
    beliefs_sets = []
    for i in range(n_beliefs+1):
        beliefs_sets.append(make_beliefs_data(n_beliefs, i))
        print("Belief set " + str(i) + " finished!")
    print("Done making belief sets!")

    print("Starting agent simulations...")
    counter = 0  # Keep track of amount of nones in each set
    lies_and_totals = []
    for beliefs_set in beliefs_sets:  # There are n_beliefs number of these (so 9)
        lies_detected = 0
        total = 0
        for cm in cm_list:  # 20 of these
            for belief in beliefs_set:  # And each of these has about 2000
                listener = agent.ListenerAgent(cm, belief, n_order, R_local, S_local, lexicon_local)
                speaker = agent.SpeakerAgent(cm, belief, n_order, R_local, S_local, lexicon_local)
                referent = np.random.choice(R_local)
                message = speaker.speak(referent)
                interpretation = listener.listen(message)
                lie_detected = listener.lie_detection(interpretation)
                if lie_detected:
                    lies_detected += 1
                total += 1
        lies_and_totals.append((lies_detected, total))
        print("Simulations for set " + str(counter) + " finished!")
        counter+=1


    for i in range(n_beliefs+1):
        lies, total = lies_and_totals[i]
        prob = lies / total
        print("For " + str(i) + " nones per belief, lie detection probability is: " + str(prob))
        print(str(lies) + "/" + str(total))

def gather_data(n_trials, none_prob, n_order):
    """
    This function makes n_trials number of random beliefs sets, with none_prob as the variable for the probability
    of a belief being none. Then for each of these random belief sets, two agents are made, a speaker and a listener,
    with n_order reasoning. The speaker speaks a lie, the listener tries to detect the lie, and it is recorder
    whether or not the lie is believed or not. The final probability of detecting the lie is printed.
    :param n_trials:
    :param none_prob:
    :param n_order:
    :return:
    """
    np.random.seed()
    n_referents = 4
    R_local = [0, 1, 2, 3]
    S_local = [0, 1, 2, 3]
    lexicon_local = [[1, 0, 0, 1],
                     [0, 1, 1, 0],
                     [1, 1, 0, 1],
                     [0, 0, 1, 1]]
    cm_local = [[0, -1, 1, 1],
                [-1, 0, 0, 1],
                [1, 0, 0, 0],
                [1, 1, 0, 0]]
    beliefs_data = old_make_beliefs_data(n_trials, n_referents, none_prob)
    lies_detected = 0
    total = 0
    for beliefs in beliefs_data:
        listener = agent.ListenerAgent(cm_local, beliefs, n_order, R_local, S_local, lexicon_local)
        speaker = agent.SpeakerAgent(cm_local, beliefs, n_order, R_local, S_local, lexicon_local)
        referent = np.random.choice(R_local)
        message = speaker.speak(referent)
        interpretation = listener.listen(message)
        lie_detected = listener.lie_detection(interpretation)
        if lie_detected:
            lies_detected += 1
        total += 1
    detection_prob = lies_detected / total
    print("Probability of lie detected: " + str(detection_prob) + ", with probability of None: " + str(none_prob))


def old_make_beliefs_data(n_sets, n_per_set, none_prob):
    np.random.seed()
    """
    Makes random belief sets with n_per_set as number of beliefs per set, and n_sets as the number of beliefs sets
    and none_prob as the probability of a belief being None.
    :param n_sets:
    :param n_per_set:
    :return:
    """
    beliefs_data = []
    for i in range(n_sets):
        beliefs = []
        for j in [np.random.random() for i in range(n_per_set)]:
            if j <= none_prob:
                beliefs.append(None)
            elif np.random.rand() > .5:
                beliefs.append(True)
            else:
                beliefs.append(False)
        beliefs_data.append(beliefs)
    return beliefs_data


def make_beliefs_data(n_per_set, none_per_set):  # Exhaustive thing! Every possible thing!
    """
    This function makes an exhaustive list of every possible beliefs, given the number of beliefs and the amount of
    none-beliefs it should have.
    The number of sets of the output = n_per_set choose none_per set * 2^(n_per_set - none_per_set)
    :param n_per_set: number of beliefs in each beliefs set
    :param none_per_set: number of nones in each beliefs set
    :return: every possible beliefs set for the input variables
    """
    belief_data = [True for i in range(n_per_set)]
    for i in range(none_per_set):
        belief_data[i] = None

    number_of_bools = n_per_set - none_per_set
    nones_full = list(itertools.permutations(belief_data))
    nones_tuple = [i for n, i in enumerate(nones_full) if i not in nones_full[:n]]  # Removes duplicates
    bools_tuple = list(itertools.product([True, False], repeat=number_of_bools))
    nones = [list(tuple_item) for tuple_item in nones_tuple]
    bools = [list(tuple_item) for tuple_item in bools_tuple]

    all_sets = []
    for i in range(len(nones)):  # For each set of beliefs (to be made)
        for j in range(len(bools)):
            none_set = copy.deepcopy(nones[i])
            bool_set = bools[j]
            bool_index = 0
            for k in range(len(none_set)):
                if none_set[k] == True:
                    none_set[k] = bool_set[bool_index]
                    bool_index += 1
            all_sets.append(none_set)
    return all_sets


def make_random_cm(nr_of_beliefs):
    """
    This function randomly creates a coherence matrix for the number of beliefs. The side of the output is a 2D array
    of nr_of_beliefs x nr_of_beliefs. The values 0, 1 and -1 are equally likely to happen.
    :param nr_of_beliefs:
    :return: coherence matrix
    """
    np.random.seed()
    checklist = [[1 for i in range(nr_of_beliefs)] for j in
                 range(nr_of_beliefs)]  # This list is to check if the value has been
    # filled in yet
    cm = [[0 for i in range(nr_of_beliefs)] for j in range(nr_of_beliefs)]  # Initialize the coherence matrix 2D list
    possibilities = [0, 1, -1]
    for i in range(nr_of_beliefs):
        for j in range(nr_of_beliefs):
            if checklist[i][j] == 1:
                if i == j:  # There is no node connected to itself, so 0 for i = j
                    cm[i][j] = 0
                else:
                    number = np.random.choice(possibilities)  # randomly picks either 1, -1 or 0
                    cm[i][j] = number
                    cm[j][i] = number
                checklist[i][j] = 0
                checklist[j][i] = 0
    return cm


def rsa_tester(R, S, agent, n):
    """
    Tests the RSA functions and will print the values of speaker RSA for a speaker agent for signals S, referents R,
    and the agent from input with n order of reasoning.
    :param R:
    :param S:
    :param agent:
    :param n:
    :return:
    """
    for s in S:
        row = ""
        for r in R:
            row = row + str("%.2f" % agent.pr_ss(s, r, n)) + "  "
        print(row)


def main_program():
    """
    RSA validity test with Floris

    lexicon1 = [[1, 0, 0],
            [0, 1, 1],
            [1, 1, 0]]

    lexicon2 = [[0, 0, 1],
                [1, 0, 1],
                [0, 1, 1]]

    lexicon3 = [[0, 1, 1],
                [0, 1, 0],
                [1, 0, 1]]

    Rtest = [0, 1, 2]
    Stest = [0, 1, 2]

    test_agent1 = agent.SpeakerAgent(cm_s, beliefs_s, 2, Rtest, Stest, lexicon1)
    rsa_tester(Rtest, Stest, test_agent1, 2)
    print()
    test_agent2 = agent.SpeakerAgent(cm_s, beliefs_s, 2, Rtest, Stest, lexicon2)
    rsa_tester(Rtest, Stest, test_agent2, 2)
    print()
    test_agent3 = agent.SpeakerAgent(cm_s, beliefs_s, 2, Rtest, Stest, lexicon3)
    rsa_tester(Rtest, Stest, test_agent3, 2)
    print()
    """
    """
    RSA test with lexicon from Blokpoel 2019 for several settings
    lexicon1 = [[1, 0],
               [0, 1],
               [1, 1]]

    lexicon2 = [[1, 0],
                [0, 1],
                [1, 1]]

    lexicon3 = [[1, 0],
                [0, 1],
                [1, 1]]

    R = [0, 1]
    S = [0, 1, 2]

    cm_s = [[0, 1],
            [1, 0]]
    cm_l = [[0, 1],
            [1, 0]]
    beliefs_s = [True, False]
    beliefs_l = [None, None]
    n = 0

    test_agent1 = agent.SpeakerAgent(cm_s, beliefs_s, 2, R, S, lexicon1)
    rsa_tester(R, S, test_agent1, 0)
    print()
    test_agent2 = agent.SpeakerAgent(cm_s, beliefs_s, 2, R, S, lexicon2)
    rsa_tester(R, S, test_agent2, 1)
    print()
    test_agent3 = agent.SpeakerAgent(cm_s, beliefs_s, 2, R, S, lexicon3)
    rsa_tester(R, S, test_agent3, 3)
    print()
    """

    # Test the make_beliefs_data function and print results
    test_data_creation = False
    if test_data_creation:
        datas = make_beliefs_data(8, 0)
        print(datas)
        print(len(datas))
        datas = make_beliefs_data(8, 1)
        print(datas)
        print(len(datas))
        datas = make_beliefs_data(8, 2)
        print(datas)
        print(len(datas))
        datas = make_beliefs_data(8, 3)
        print(datas)
        print(len(datas))
        datas = make_beliefs_data(8, 4)
        print(datas)
        print(len(datas))
        datas = make_beliefs_data(8, 5)
        print(datas)
        print(len(datas))
        datas = make_beliefs_data(8, 6)
        print(datas)
        print(len(datas))
        datas = make_beliefs_data(8, 7)
        print(datas)
        print(len(datas))
        datas = make_beliefs_data(8, 8)
        print(datas)
        print(len(datas))

    # Runs the main data gathering code if it is set to True.
    gather_new_data = True
    if gather_new_data:
        new_gather_data(8, 1)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main_program()
