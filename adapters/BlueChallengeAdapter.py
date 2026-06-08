from adapters.BaseAdapter import BaseAdapter
from adapters.OpenAIGymAdapter import OpenAIGymAdapter
from adapters.BlueTableAdapter import BlueTableAdapter
from adapters.EnumActionAdapter import EnumActionAdapter
from CybORG.Shared.Results import Results


class BlueChallengeAdapter(BaseAdapter):
    def __init__(
        self,
        agent_name: str,
        env: BaseAdapter,
        sample_result: Results,
        agent=None,
        reward_threshold=None,
        max_steps=None,
    ):
        super().__init__(env, agent)
        self.agent_name = agent_name
        env = BlueTableAdapter(env, output_mode="vector")
        env = EnumActionAdapter(env)
        env = OpenAIGymAdapter(
            agent_name=agent_name, env=env, sample_result=sample_result
        )

        self.env = env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None

    def pre_reset(self):
        self.step_counter = 0
        return self.env.pre_reset()

    def post_reset(self, result):
        return self.env.post_reset(result)

    def pre_step(self, action):
        return self.env.pre_step(action)

    def post_step(self, action, result):
        result = self.env.post_step(action, result)
        self.step_counter += 1
        return result

    def get_attr(self, attribute: str):
        return self.env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self.env.get_observation(agent)

    def get_agent_state(self, agent: str):
        return self.env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self.env.get_action_space(self.agent_name)

    def get_last_action(self, agent):
        return self.get_attr("get_last_action")(agent)

    def get_ip_map(self):
        return self.get_attr("get_ip_map")()

    def get_rewards(self):
        return self.get_attr("get_rewards")()

    def get_reward_breakdown(self, agent: str):
        return self.get_attr("get_reward_breakdown")(agent)
