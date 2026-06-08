import inspect
from typing import Union

from CybORG.Agents.SimpleAgents import BaseAgent

from adapters.BaseAdapter import BaseAdapter


class EnumActionAdapter(BaseAdapter):
    def __init__(self, env: Union[type, BaseAdapter] = None, agent: BaseAgent = None):
        super().__init__(env, agent)
        self.possible_actions = None
        self.action_signature = {}
        self.get_action_space("Blue")

    def pre_reset(self):
        return self.env.pre_reset()

    def post_reset(self, result):
        return self.env.post_reset(result)

    def pre_step(self, action):
        if action is not None:
            action = self.possible_actions[action]
        return action

    def post_step(self, action, result):
        result = self.env.post_step(action, result)
        result.action_space = self.action_space_change(result.action_space)
        return result

    def action_space_change(self, action_space: dict) -> int:
        assert type(action_space) is dict, (
            "Wrapper required a dictionary action space. "
            "Please check that the wrappers below the ReduceActionSpaceWrapper return the action space as a dict "
        )
        possible_actions = []
        temp = {}
        params = ["action"]
        # for action in action_space['action']:
        for i, action in enumerate(action_space["action"]):
            if action not in self.action_signature:
                self.action_signature[action] = inspect.signature(action).parameters
            # param_dict = {}
            param_list = [{}]
            for p in self.action_signature[action]:
                if p == "priority":
                    continue
                temp[p] = []
                if p not in params:
                    params.append(p)

                if len(action_space[p]) == 1:
                    for p_dict in param_list:
                        p_dict[p] = list(action_space[p].keys())[0]
                else:
                    new_param_list = []
                    for p_dict in param_list:
                        for key, val in action_space[p].items():
                            p_dict[p] = key
                            new_param_list.append(
                                {key: value for key, value in p_dict.items()}
                            )
                    param_list = new_param_list
            for p_dict in param_list:
                possible_actions.append(action(**p_dict))

        self.possible_actions = possible_actions
        return len(possible_actions)
