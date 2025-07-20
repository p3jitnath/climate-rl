import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from gymnasium.utils import seeding


class SimpleClimateBiasCorrectionEnv(gym.Env):
    """
    A gym environment for a simple climate bias correction problem.

    The environment simulates a single temperature variable that evolves over time, with a goal
    to learn a heating increment (`u`) which minimizes the size of an analysis bias correction term.

    Attributes:
        min_temperature (float): Minimum normalized temperature.
        max_temperature (float): Maximum normalized temperature.
        max_heating_rate (float): Maximum heating rate.
        dt (float): Time step in minutes.
        count (float): Counter for internal use.
        viewer: For rendering the environment.
        action_space (gym.spaces.Box): The space of possible actions (heating rates).
        observation_space (gym.spaces.Box): The space of possible states (temperatures).
        np_random: Random number generator for the environment.

    Author:
        Mark Webb, Met Office
        Email: mark.webb@metoffice.gov.uk
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, render_mode=None):
        self.min_temperature = 0.0
        self.max_temperature = 1.0
        self.min_heating_rate = -1.0
        self.max_heating_rate = 1.0
        self.dt = 1.0  # Time step (minutes)
        self.count = 0.0
        self.screen = None
        self.clock = None

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=self.min_heating_rate,
            high=self.max_heating_rate,
            shape=(1,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.array([self.min_temperature], dtype=np.float32),
            high=np.array([self.max_temperature], dtype=np.float32),
            dtype=np.float32,
        )

        # Check render_mode is valid
        assert (
            render_mode is None or render_mode in self.metadata["render_modes"]
        )
        self.render_mode = render_mode

    def step(self, u):
        """
        Performs one step in the environment using the action `u`.

        Args:
            u (float): The action, representing a heating increment.

        Returns:
            tuple: A tuple containing the new observation, the reward, whether the episode is done,
                   and additional information.
        """
        current_temperature = self.state[0]

        # Clip action to the allowed range
        u = np.clip(u, -self.max_heating_rate, self.max_heating_rate)[0]

        # Calculate new temperature
        observed_temperature = (321.75 - 273.15) / 100
        physics_temperature = (380 - 273.15) / 100
        division_constant = physics_temperature - observed_temperature

        new_temperature = current_temperature + u
        relaxation = (
            (physics_temperature - current_temperature)
            * 0.2
            / division_constant
        )
        new_temperature += relaxation

        # OLD - v0
        # bias_correction = (
        #     (observed_temperature - new_temperature)
        #     * 0.1
        #     / division_constant
        # )
        # new_temperature += bias_correction

        # OLD - v0
        # Calculate cost (negative reward)
        # costs = bias_correction ** 2

        # NEW - v1
        # Calculate cost (mean squared error)
        costs = (observed_temperature - new_temperature) ** 2

        new_temperature = np.clip(
            new_temperature, self.min_temperature, self.max_temperature
        )

        self.state = np.array([new_temperature])

        return self._get_obs(), -costs, False, False, self._get_info()

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.

        Returns:
            np.array: The initial observation.
        """
        super().reset(seed=seed)
        self.state = np.array([(300 - 273.15) / 100])  # Starting temperature

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def _get_info(self):
        return {"_": None}

    def _get_obs(self):
        """
        Returns the current observation (state).

        Returns:
            np.array: The current observation.
        """
        temperature = self.state[0]
        return np.array([temperature], dtype=np.float32)

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Render the environment's current state to a window.

        This method visualizes the current temperature state of the environment using a simple thermometer
        representation. The mercury level in the thermometer increases or decreases in accordance with the
        current normalized temperature value, providing a visual indication of temperature changes over time.

        Returns:
            np.ndarray or None
                - If self.render_mode is 'rgb_array', returns an RGB array of the screen.
                - If self.render_mode is 'human', returns None.
        """

        screen_width = 600
        screen_height = 400
        thermometer_height = 300
        thermometer_width = 50
        mercury_width = 30
        base_height = 10

        temp_range = self.max_temperature - self.min_temperature
        mercury_height = (
            (self.state[0] - self.min_temperature) / temp_range
        ) * thermometer_height

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode(
                    (screen_width, screen_height)
                )
            else:  # For rgb_array render mode, we don't need to display the window
                self.screen = pygame.Surface((screen_width, screen_height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)  # Initialize the font

        self.screen.fill((255, 255, 255))  # Fill the background with white

        # Draw Thermometer
        thermometer_rect = pygame.Rect(
            (screen_width / 2) - (thermometer_width / 2),
            (screen_height / 2) - (thermometer_height / 2),
            thermometer_width,
            thermometer_height,
        )
        pygame.draw.rect(
            self.screen, (200, 200, 200), thermometer_rect
        )  # Light gray

        # Draw Mercury
        mercury_rect = pygame.Rect(
            (screen_width / 2) - (mercury_width / 2),
            (screen_height / 2) + (thermometer_height / 2) - mercury_height,
            mercury_width,
            mercury_height,
        )
        pygame.draw.rect(self.screen, (255, 0, 0), mercury_rect)  # Red

        # Draw Base
        base_rect = pygame.Rect(
            (screen_width / 2) - (thermometer_width / 2),
            (screen_height / 2) + (thermometer_height / 2),
            thermometer_width,
            base_height,
        )
        pygame.draw.rect(self.screen, (150, 150, 150), base_rect)  # Dark gray

        # Calculate the position for the observed mark line
        observed_ratio = (321.75 - 273.15) / (380 - 273.15)
        observed_mark_y = (screen_height / 2) + (thermometer_height / 2)
        observed_mark_y -= thermometer_height * observed_ratio

        observed_mark_start = (screen_width / 2) - (thermometer_width / 2)
        observed_mark_end = (screen_width / 2) + (thermometer_width / 2)

        # Draw the observed mark line
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (observed_mark_start, observed_mark_y),
            (observed_mark_end, observed_mark_y),
            5,
        )  # Black line

        # Draw temperature markings every x degrees from 273.15 K to 380 K
        min_temp_k = 273.15
        max_temp_k = 380
        temp_range_k = max_temp_k - min_temp_k
        marking_spacing_k = 20  # Every x degrees

        for temp_k in range(
            int(min_temp_k), int(max_temp_k) + 1, marking_spacing_k
        ):
            # Normalize the temperature to [0, 1] for the current scale
            normalized_temp = (temp_k - min_temp_k) / temp_range_k
            # Calculate the Y position for the marking based on the normalized temperature
            mark_y = (
                (screen_height / 2)
                + (thermometer_height / 2)
                - (normalized_temp * thermometer_height)
            )

            # Draw the marking line
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                ((screen_width / 2) - (thermometer_width / 2) - 10, mark_y),
                ((screen_width / 2) - (thermometer_width / 2), mark_y),
                2,
            )

            # Render the temperature text
            temp_text = self.font.render(f"{temp_k} K", True, (0, 0, 0))
            self.screen.blit(
                temp_text,
                (
                    (screen_width / 2) - (thermometer_width / 2) - 60,
                    mark_y - 10,
                ),
            )

        # Display Thermometer
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return pygame.surfarray.array3d(self.screen).transpose([1, 0, 2])

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
