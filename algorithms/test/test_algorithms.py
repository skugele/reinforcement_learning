from unittest import TestCase

from algorithms import Policy, calculate_return, generate_episode
from algorithms import Action
from algorithms import State
from algorithms import ValueFunction

import collections

from random import randint, choice


class TestPolicy(TestCase):
    def test_add_actions_increases_length_of_actions_for_state(self):
        policy = Policy()

        s1 = State('1')

        actions = [Action('1'), Action('2'), Action('3'), Action('4')]
        p = [0.9, 0.05, 0.03, 0.02]

        policy.add_actions(state=s1, actions=actions, p=p)

        self.assertEqual(len(policy[s1]), len(actions), msg='Length does not match excepted length of 4')

        for a, p in zip(actions, p):
            self.assertEqual(policy[s1][a], p, msg='Probability for action is incorrect')

    def test_add_actions_to_multiple_states(self):
        policy = Policy()

        first = {'state': State('1'),
                 'actions': [Action('1'), Action('2'), Action('3'), Action('4')],
                 'p': [0.9, 0.05, 0.03, 0.02]}

        second = {'state': State('2'),
                  'actions': [Action('1'), Action('2'), Action('3')],
                  'p': [0.75, 0.15, 0.1]}

        policy.add_actions(state=first['state'], actions=first['actions'], p=first['p'])
        policy.add_actions(state=second['state'], actions=second['actions'], p=second['p'])

        self.assertEqual(len(policy[first['state']]), len(first['actions']),
                         msg='Length does not match excepted length')
        self.assertEqual(len(policy[second['state']]), len(second['actions']),
                         msg='Length does not match excepted length')

        for a, p in zip(first['actions'], first['p']):
            self.assertEqual(policy[first['state']][a], p, msg='Probability for action is incorrect')

        for a, p in zip(second['actions'], second['p']):
            self.assertEqual(policy[second['state']][a], p, msg='Probability for action is incorrect')

    def test_add_actions_to_policy_with_illegal_probability_sum_raises_exception(self):
        policy = Policy()

        s1 = State('1')
        actions = [Action('1'), Action('2'), Action('3'), Action('4')]

        # Sum greater than 1
        illegal_prob1 = [1.9, 0.05, 0.03, 0.02]

        with self.assertRaises(ValueError):
            policy.add_actions(state=s1, actions=actions, p=illegal_prob1)

        # Sum less than 1
        illegal_prob2 = [0.5, 0.05, 0.03, 0.02]

        with self.assertRaises(ValueError):
            policy.add_actions(state=s1, actions=actions, p=illegal_prob2)

    def test_sample_from_policy_returns_actions_for_state(self):
        policy = Policy()

        first = {'state': State('1'),
                 'actions': [Action('1'), Action('2'), Action('3'), Action('4')],
                 'p': [0.9, 0.05, 0.03, 0.02]}

        second = {'state': State('2'),
                  'actions': [Action('1'), Action('2'), Action('3')],
                  'p': [0.75, 0.15, 0.1]}

        policy.add_actions(first['state'], actions=first['actions'], p=first['p'])
        policy.add_actions(second['state'], actions=second['actions'], p=second['p'])

        self.assertIn(policy.sample(first['state']), first['actions'],
                      msg='Sample returned an action not associated with this state')
        self.assertIn(policy.sample(second['state']), second['actions'],
                      msg='Sample returned an action not associated with this state')

    def test_sample_n_actions_from_policy_returns_correct_distribution(self):
        policy = Policy()

        first = {'state': State('1'),
                 'actions': [Action('1'), Action('2'), Action('3'), Action('4')],
                 'p': [0.9, 0.05, 0.03, 0.02]}

        second = {'state': State('2'),
                  'actions': [Action('1'), Action('2'), Action('3')],
                  'p': [0.75, 0.15, 0.1]}

        policy.add_actions(state=first['state'], actions=first['actions'], p=first['p'])
        policy.add_actions(state=second['state'], actions=second['actions'], p=second['p'])

        n_samples = 10000
        c = collections.Counter(policy.sample_n(first['state'], n=n_samples))

        for a, p in zip(first['actions'], first['p']):
            self.assertAlmostEqual(c[a] / n_samples, p, delta=0.05)

        c = collections.Counter(policy.sample_n(second['state'], n=n_samples))

        for a, p in zip(second['actions'], second['p']):
            self.assertAlmostEqual(c[a] / n_samples, p, delta=0.05)

    def test_add_actions_with_no_probability_gives_uniform_distribution(self):
        policy = Policy()

        values = {'state': State('1'),
                  'actions': [Action('1'), Action('2'), Action('3')]}

        policy.add_actions(values['state'], actions=values['actions'])

        n_samples = 10000
        c = collections.Counter(policy.sample_n(values['state'], n=n_samples))

        for a in values['actions']:
            expected = 1.0 / len(values['actions'])
            actual = c[a] / n_samples
            self.assertAlmostEqual(expected, actual, delta=0.05)


class TestCalculateReturn(TestCase):
    def test_calculate_return_is_zero_when_empty_list(self):
        self.assertEqual(calculate_return(rewards=[]), 0.0)

    def test_calculate_return_when_undiscounted(self):
        r = [1] * 100
        self.assertEqual(calculate_return(rewards=r), sum(r))

    def test_calculate_return_raises_error_when_discount_invalid(self):
        # Test discount < 0.0
        with self.assertRaises(ValueError):
            calculate_return(rewards=[1] * 10, discount=-0.01)

        # Test discount > 1.0
        with self.assertRaises(ValueError):
            calculate_return(rewards=[1] * 10, discount=1.0001)

    def test_calculate_return_when_maximal_discount(self):
        r = [17, 1e4, 1e5]
        self.assertEqual(calculate_return(rewards=r, discount=0.0), r[0])

    def test_calculate_return_when_non_maximal_discount(self):
        r = [2 ** x for x in range(10)]
        self.assertEqual(calculate_return(rewards=r, discount=0.5), len(r))


class TestGenerateEpisode(TestCase):
    def setUp(self):
        self.actions = [Action(n) for n in range(4)]
        self.states = [State(n) for n in range(100)]

    def test_generate_episode(self):
        start_state = choice(self.states)
        term_state = choice(self.states)

        episode = generate_episode(actions=self.actions,
                                   states=self.states,
                                   reward_func=lambda x, y, z: randint(0, 10),
                                   start_state=start_state,
                                   term_states=[term_state])

        first = last = next(episode, '')
        for last in episode:
            print(last)

        self.assertEqual(first[0], start_state, msg='Initial transition does not start with start state!')
        self.assertEqual(last[3], term_state, msg='Final transition does not end with terminal state!')


class TestValueFunction(TestCase):
    def test_value_function_default_values(self):
        q = ValueFunction(default=0.1)

        self.assertEqual(q(), 0.1)
        self.assertEqual(q(Action(1), State(1)), 0.1)

    def test_update_value_in_value_function(self):
        q = ValueFunction(default=0.1)

        q.update(params=(Action(1), State(1)), value=5.0)
        self.assertEqual(q(Action(1), State(1)), 5.0)
        self.assertEqual(q(Action(2), State(1)), 0.1)
