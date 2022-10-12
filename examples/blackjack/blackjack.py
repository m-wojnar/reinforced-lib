from functools import reduce
from typing import Dict, Tuple

import gym
import numpy as np


gym.envs.registration.register(
    id='BlackjackEnv-v0',
    entry_point='examples.blackjack.main:BlackjackEnv'
)


class BlackjackEnv(gym.Env):

    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({
            "current_sum": gym.spaces.Discrete(22), # {0, 2, 3, ..., 22}
            "dealers_card": gym.spaces.Discrete(6), # 9-Ace
        })

        self.cards_values = np.array([0, 10, 2, 3, 4, 11])
        self.cards_names = np.array(["9", "10", "Jack", "Queen", "King", "Ace"])
    
    def _draw_card(self):
        return self.deck.pop()
    
    def _count_points(self):
        return np.sum(self.cards_values[self.players_hand])
    
    def _on_hit(self, render):

        self.players_hand.append(self._draw_card())
        done = self._count_points() > 21
        reward = -1 if done else 0

        self.render() if render else None

        return {
            "current_sum": self._count_points(),
            "dealers_card": self.dealers_card
        }, reward, done, False, {}

    def _on_stick(self, render):

            # We first need to simulate the dealers play:
        
        # Assume the dealer takes the rest of the deck
        dealers_order = np.concatenate((self.dealers_hand, self.deck))

        # Translate dealers new hand to points
        dealers_order = self.cards_values[dealers_order]

        # Dealer hits with 17s stradegy
        hit = lambda acc, x: acc + x if acc < 17 else acc
        dealers_score = reduce(hit, dealers_order, 0)

            # We now must determine the winner:

        reward = 1
        if dealers_score <= 21:
            if self._count_points() == dealers_score:
                reward = 0
            elif self._count_points() > dealers_score:
                reward = 1
            else:
                reward = -1
        
        self.render() if render else None

        return {
            "current_sum": self._count_points(),
            "dealers_card": self.dealers_card
        }, reward, True, False, {}
    
    def reset(self, seed: int = None):

        seed = seed if seed else np.random.randint(1000)
        super().reset(seed=seed)

        self.deck = list(np.random.permutation(24) % 6)
        self.players_hand = [self._draw_card(), self._draw_card()]
        self.dealers_hand = [self._draw_card(), self._draw_card()]
        self.dealers_card = self.dealers_hand[0]

        return {
            "current_sum": self._count_points(),
            "dealers_card": self.dealers_card
        }, {}
    
    def step(self, action: int, render: bool = False) -> Tuple[gym.spaces.Dict, float, bool, bool, Dict]:

        return self._on_hit(render) if action else self._on_stick(render)
    
    def render(self):

        players_hand = f"{self.cards_names[self.players_hand]}"
        dealers_hand = f"<{self.cards_names[self.dealers_card]}> {self.cards_names[self.dealers_hand[1:]]}"
        lines = "-"*32

        print(f"{lines}\nPlayer: {players_hand} (sum={self._count_points()})\nDealer: {dealers_hand}\n{lines}")
        
        