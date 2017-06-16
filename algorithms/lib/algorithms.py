import numpy
import collections

from random import choice


class State(object):
    def __init__(self, label):
        self.label = label

    def __eq__(self, other):
        if self.label == other.label:
            return True
        return False

    def __repr__(self):
        return str(self.label)

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if not isinstance(other, State):
            return False

        return self.label == other.label


class Action(object):
    def __init__(self, label):
        self.label = label

    def __eq__(self, other):
        if self.label == other.label:
            return True
        return False

    def __repr__(self):
        return str(self.label)

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if not isinstance(other, Action):
            return False

        return self.label == other.label


StateAction = collections.namedtuple('StateAction', ['state', 'action'])


class Policy(object):
    PROBABILITY_TOLERANCE_THRESHOLD = 1e-3

    def __init__(self):
        self.states = dict()

    def sample(self, state):
        action_dist = self.states[state]
        return numpy.random.choice(a=list(action_dist.keys()),
                                   p=list(action_dist.values()))

    def sample_n(self, state, n):
        action_dist = self.states[state]

        num = 0
        while num < n:
            yield numpy.random.choice(a=list(action_dist.keys()),
                                      p=list(action_dist.values()))
            num += 1

    def add_actions(self, state, actions, p=None):
        if p is None:
            p = [1.0 / len(actions)] * len(actions)

        if numpy.abs(1.0 - numpy.sum(p)) > self.PROBABILITY_TOLERANCE_THRESHOLD:
            raise ValueError('Probabilities must equal 1')

        self.states[state] = dict(zip(actions, p))

    def __getitem__(self, item):
        return self.states[item]


# Control Algorithms
class GeneralizedPolicyIteration(object):
    def __init__(self):
        pass

    def evaluate(self):
        pass

    def improve(self):
        pass

    def iterate(self):
        pass


class Sarsa(object):
    def __init__(self):
        pass

    def next_q(self, sa, next_sa, reward, alpha, gamma, q):
        return q(sa) + alpha * (reward + gamma * q(next_sa) - q(sa))


class ValueFunction(object):
    def __init__(self, default=0.0):
        self.default = default

        self._values = dict()

    # Q(A, S) = Q(A, S) + alpha * (R + gamma * Q(A_next, S_next) - Q(A, S))
    def __call__(self, *params, **param_keywords):
        return self._values.get(params, self.default)

    # Q.set(params=(A, S), new_value=)
    def update(self, params, value):
        self._values[params] = value


# Helper Functions
def calculate_return(rewards, discount=1.0):
    if not (0.0 <= discount <= 1.0):
        raise ValueError('Discount must be between 0.0 and 1.0 (inclusive)')

    value = 0.0
    for step, r in enumerate(rewards, start=0):
        value += (discount ** step) * r

    return value


Transition = collections.namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


# A function used for testing purposes only
def generate_episode(actions, states, reward_func, transition_func=None, term_states=None, start_state=None):
    # Choose an initial state randomly if not provided
    state = start_state or choice(states)

    while state not in term_states:
        action = choice(actions)

        next_state = None
        if transition_func:
            next_state = transition_func(state, action)
        else:
            next_state = choice(states)

        reward = reward_func(state, action, next_state)

        yield Transition(state, action, reward, next_state)
        state = next_state
