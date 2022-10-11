import random
import sys
from typing import Callable, Dict, Tuple

import gym
import jax
import jax.numpy as jnp
from chex import dataclass, Array, Numeric, PRNGKey, Scalar


gym.envs.registration.register(
    id='BlackjackEnv',
    entry_point='examples.blackjack.game:BlackjackEnv'
)


class BlackjackEnv(gym.Env):

    def __init__(self) -> None:
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({
            "current_sum": gym.spaces.Discrete(23), # 0-22
            "dealers_card": gym.spaces.Discrete(6), # 9-Ace
        })

        self.cards_values = jnp.array([0, 10, 2, 3, 4, 11])
        self.cards_names = ["9", "10", "Jack", "Queen", "King", "Ace"]
    
    def __draw_card(self):
        return (self.deck.pop() % 6).astype(int)
    
    def __count_points(self, hand):
        return jnp.sum(jnp.take(self.cards_values, jnp.array(hand)))
    
    def __on_hit(self):

        self.players_hand.append(self.__draw_card())
        done = self.__count_points(self.players_hand) > 21
        reward = jax.lax.cond(done, lambda: -1, lambda: 0)

        return {
            "current_sum": self.__count_points(self.players_hand),
            "dealers_card": self.dealers_card
        }, reward, done, {}

    def __on_stick(self):

            # We first need to simulate the dealers play:
        
        # Assume the dealer takes the rest of the deck
        dealers_order = self.dealers_hand + list(map(lambda c: c % 6, self.deck))

        # Translate dealers new hand to points
        dealers_order = jnp.take(self.cards_values, jnp.array(dealers_order))

        # Calculate the trace of dealers gameplay
        scan_fun = lambda c, x: (c + x, c + x)
        _, dealers_order = jax.lax.scan(scan_fun, 0, dealers_order)

        # Dealers last move was the first card which resulted in sum > 16
        dealers_score = dealers_order[jnp.where(dealers_order > 16, size=1)][0]

            # We now must determine the winner:

        # if dealers_score > 21:
        #     reward = 1
        # else:
        #     if self.__count_points(self.players_hand) == dealers_score:
        #         reward = 0
        #     else:
        #         if self.__count_points(self.players_hand) > dealers_score:
        #             reward = -1
        #         else:
        #             reward = 0
        
        def win():
            return 1
        
        def draw():
            return 0
        
        def lose():
            return -1
        
        def compare_strict():
            return jax.lax.cond(self.__count_points(self.players_hand) > dealers_score, win, lose)
        
        def compare():
            return jax.lax.cond(self.__count_points(self.players_hand) == dealers_score, draw, compare_strict)
        
        reward = jax.lax.cond(dealers_score > 21, win, compare)

        return {
            "current_sum": self.__count_points(self.players_hand),
            "dealers_card": self.dealers_card
        }, reward, True, {}
    
    def reset(self, seed: int = None):

        seed = seed if seed else random.randint(0, sys.maxsize)
        super().reset(seed=seed)
        self.key = jax.random.PRNGKey(seed)

        self.deck = list(jax.random.permutation(self.key, 24))
        self.players_hand = [self.__draw_card(), self.__draw_card()]
        self.dealers_hand = [self.__draw_card(), self.__draw_card()]
        self.dealers_card = self.dealers_hand[0]

        return {
            "current_sum": self.__count_points(self.players_hand),
            "dealers_card": self.dealers_card
        }, {}
    
    def step(self, action: int):
        return jax.lax.cond(action == 1, self.__on_hit, self.__on_stick)
    
    def render(self):

        players_hand = f"{list(map(lambda i: self.cards_names[i], self.players_hand))}"
        dealers_hand = f"<{self.cards_names[self.dealers_card]}> {list(map(lambda i: self.cards_names[i], self.players_hand))[1:]}"
        curr_sum = self.__count_points(self.players_hand)
        lines = "-"*32

        print(f"{lines}\nPlayer: {players_hand} (sum={curr_sum})\nDealer: {dealers_hand}\n{lines}")
        
        