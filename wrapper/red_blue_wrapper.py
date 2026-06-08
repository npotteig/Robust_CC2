from typing import Union
import copy

from CybORG import CybORG
from CybORG.Shared.Actions import Action, Sleep

from adapters.BaseAdapter import Adapter


class RedBlueWrapper:
    def __init__(self, env: CybORG, red_adapter: Adapter, blue_adapter: Adapter):
        self.env = env
        self.red_adapter = red_adapter
        self.blue_adapter = blue_adapter
        self.agent_names = ["Red", "Blue"]

    def reset(self):
        self.red_adapter.pre_reset()
        self.blue_adapter.pre_reset()
        result = self.env.reset_parallel(self.agent_names)
        blue_real_obs = copy.deepcopy(result["Blue"].observation)
        blue_real_action_space = copy.deepcopy(result["Blue"].action_space)
        red_obs, red_info = self.red_adapter.post_reset(result["Red"])
        blue_obs, blue_info = self.blue_adapter.post_reset(result["Blue"])
        blue_info["observation"] = blue_real_obs
        blue_info["action_space"] = blue_real_action_space
        return {"Red": red_obs, "Blue": blue_obs}, {"Red": red_info, "Blue": blue_info}

    def step(
        self,
        red_action_int: Union[int, Action],
        blue_action_int: int,
        green_action: Action = None,
    ):
        red_action_cyborg = self.red_adapter.pre_step(red_action_int)
        blue_action_cyborg = (
            self.blue_adapter.pre_step(blue_action_int)
            if isinstance(blue_action_int, int)
            else blue_action_int
        )
        green_action_cyborg = green_action if green_action is not None else Sleep()
        # print("Step")
        # print("red action cyborg:", red_action_cyborg[-1])
        # print("red action cyborg:", [f"{red_action}" for red_action in red_action_cyborg])
        # print("blue action cyborg:", blue_action_cyborg)
        result = self.env.step_parallel(
            {
                "Red": red_action_cyborg,
                "Blue": blue_action_cyborg,
                "Green": green_action_cyborg,
            }
        )
        blue_real_obs = copy.deepcopy(result["Blue"].observation)
        blue_real_action_space = copy.deepcopy(result["Blue"].action_space)
        red_obs, red_reward, red_done, red_truncated, red_info = (
            self.red_adapter.post_step(
                red_action_cyborg[-1], copy.deepcopy(result["Red"])
            )
        )
        blue_obs, blue_reward, blue_done, blue_truncated, blue_info = (
            self.blue_adapter.post_step(blue_action_int, copy.deepcopy(result["Blue"]))
        )
        blue_info["observation"] = blue_real_obs
        blue_info["action_space"] = blue_real_action_space
        blue_info["cyborg_action"] = blue_action_cyborg
        return (
            {"Red": red_obs, "Blue": blue_obs},
            {"Red": red_reward, "Blue": blue_reward},
            {"Red": red_done, "Blue": blue_done},
            {"Red": red_truncated, "Blue": blue_truncated},
            {"Red": red_info, "Blue": blue_info},
        )
