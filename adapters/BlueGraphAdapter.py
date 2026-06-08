import torch
from adapters.BlueTableAdapter import BlueTableAdapter
from adapters.BlueChallengeAdapter import BlueChallengeAdapter
from adapters.EnumActionAdapter import EnumActionAdapter
from adapters.OpenAIGymAdapter import OpenAIGymAdapter
from CybORG.Shared.Actions.Action import Sleep
from CybORG.Shared.Results import Results

from wrapper.observation_graph import ObservationGraph


class BlueGraphAdapter(BlueChallengeAdapter):
    def __init__(
        self,
        agent_name: str,
        env,
        sample_result: Results,
        agent=None,
        reward_threshold=None,
        max_steps=None,
    ):
        super().__init__(
            agent_name, env, sample_result, agent, reward_threshold, max_steps
        )
        self.agent_name = agent_name

        self.original_env = env
        env = BlueTableAdapter(env, output_mode="vector")
        env = EnumActionAdapter(env)
        env.get_action_space("Blue")
        self.possible_actions = env.possible_actions
        env = OpenAIGymAdapter(
            agent_name=agent_name, env=env, sample_result=sample_result
        )

        self.graph = ObservationGraph()

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None
        self.last_obs = None

    def pre_reset(self):
        # self.phase = 1
        self.prev_tabular_x = None
        return super().pre_reset()

    def post_reset(self, result):
        tab_x, info = super().post_reset(result)
        self.graph = ObservationGraph()
        self.graph.setup(self.original_env.get_observation("Blue"))
        graph_x, ei, names = self.graph.get_state(include_names=True)
        x = self._combine_data(graph_x, tab_x)
        self.last_obs = (x, ei)
        return (x, ei), {}

    def pre_step(self, action=None):
        return super().pre_step(action)

    def post_step(self, action: int, result):
        obs, reward, done, truncated, info = super().post_step(action, result)
        self.possible_actions = self.env.env.possible_actions
        if isinstance(action, int):
            action = Sleep() if action is None else self.possible_actions[action]
        dict_obs = self.original_env.get_observation(self.agent_name)
        # success = dict_obs.get('success', 'UNK')
        self.graph.parse_observation(action, dict_obs)

        x, ei, names = self.graph.get_state(include_names=True)
        # print(names)
        x = self._combine_data(x, obs)
        self.last_obs = (x, ei)

        return (x, ei), reward, done, truncated, info

    def _combine_data(self, graph_x, tabular_x):
        tabular_x = tabular_x.reshape(tabular_x.shape[0] // 4, 4)

        additional_data = torch.zeros(graph_x.size(0), 4)
        additional_data[graph_x[:, 0] == 1] = torch.tensor(
            tabular_x, dtype=torch.float32
        )
        return torch.cat([graph_x, additional_data], dim=1)
