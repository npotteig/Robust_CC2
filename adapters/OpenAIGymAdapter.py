import numpy as np
from gymnasium import spaces

from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared.Results import Results

from adapters.BaseAdapter import BaseAdapter


class OpenAIGymAdapter(BaseAdapter):
    def __init__(
        self,
        agent_name: str,
        sample_result: Results,
        env: BaseAdapter = None,
        agent: BaseAgent = None,
    ):
        super().__init__(env, agent_name)

        self.agent_name = agent_name
        if isinstance(self.get_action_space(self.agent_name), list):
            self.action_space = spaces.MultiDiscrete(
                self.get_action_space(self.agent_name)
            )
        else:
            assert isinstance(self.get_action_space(self.agent_name), int)
            self.action_space = spaces.Discrete(self.get_action_space(self.agent_name))
        box_len = len(self.observation_change(sample_result.observation))
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(box_len,), dtype=np.float32
        )
        self.reward_range = (float("-inf"), float("inf"))
        self.metadata = {}
        self.action = None

    def pre_reset(self):
        return self.env.pre_reset()

    def post_reset(self, result):
        result = self.env.post_reset(result)
        result.action_space = self.action_space_change(result.action_space)
        result.observation = self.observation_change(result.observation)
        return np.array(result.observation), {}

    def pre_step(self, action):
        self.action = action
        return self.env.pre_step(action)

    def post_step(self, action, result):
        result = self.env.post_step(action, result)
        result.observation = self.observation_change(result.observation)
        result.action_space = self.action_space_change(result.action_space)
        self.update_spaces(result)
        info = vars(result)
        return np.array(result.observation), result.reward, result.done, False, info

    def update_spaces(self, sample_result):
        if isinstance(self.get_action_space(self.agent_name), list):
            self.action_space = spaces.MultiDiscrete(
                self.get_action_space(self.agent_name)
            )
        else:
            assert isinstance(self.get_action_space(self.agent_name), int)
            self.action_space = spaces.Discrete(self.get_action_space(self.agent_name))
        box_len = len(self.observation_change(sample_result.observation))
        self.observation_space = spaces.Box(
            -1.0, 1.0, shape=(box_len,), dtype=np.float32
        )

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.get_attr("get_agent_state")(agent)

    def get_action_space(self, agent):
        return self.env.get_action_space(agent)

    def get_last_action(self, agent):
        return self.get_attr("get_last_action")(agent)

    def get_ip_map(self):
        return self.get_attr("get_ip_map")()

    def get_rewards(self):
        return self.get_attr("get_rewards")()
