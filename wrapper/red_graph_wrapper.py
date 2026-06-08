import copy
from typing import Union, Optional, Dict, List

import torch

from wrapper.red_observation_graph import RedObservationGraph
from wrapper.red_observation_graph import CompromiseLevel, HostType
from CybORG.Shared.Actions import Action, Impact
from rewards.bt import BehaviourTreeReward, BehaviourTreeConfig

from CybORG import CybORG


class RedGraphWrapper:
    def __init__(
        self,
        env: CybORG,
        max_nodes=None,
        reward_mode: str = "environment",
        bt_config: Optional[BehaviourTreeConfig] = None,
        action_mask_dict: Optional[Dict[str, List[int]]] = None,
    ):
        self.agent_name = "Red"
        self.original_env = env
        self.graph = RedObservationGraph(max_nodes=max_nodes)
        self.reward_mode = reward_mode
        self.action_mask_dict = action_mask_dict

        match reward_mode:
            case "environment":
                self.bt_config = None
            case "proc_as_reward":
                assert bt_config is not None, (
                    "BehaviourTreeConfig must be provided for proc_as_reward reward mode"
                )
                assert bt_config.use_memory, (
                    "Memory must be enabled for proc_as_reward reward mode"
                )
                self.bt_config = bt_config
            case "bt_as_reward":
                assert bt_config is not None, (
                    "BehaviourTreeConfig must be provided for bt_as_reward reward mode"
                )
                self.bt_config = bt_config
            case _:
                raise ValueError(f"Invalid reward mode: {self.reward_mode}")

    def reset(self):
        init_observation = self.original_env.reset_parallel(agents=[self.agent_name])[
            self.agent_name
        ]
        self.graph = RedObservationGraph(max_nodes=self.graph.max_nodes)
        self.graph.setup(copy.deepcopy(init_observation.observation))
        x, ei = self.graph.get_state()

        self.mission_str = "Impact OpServer"
        self.phase = 0
        if self.bt_config is not None:
            self.bt = BehaviourTreeReward.create_bt(
                mission_str=self.mission_str,
                bt_config=self.bt_config,
                action_masks=self.action_mask_dict,
            )
            _, self.action_mask, self.phase = self.bt.step_reward(
                state=(x.numpy(), ei.numpy()), mission_str=self.mission_str
            )
        # convert phase to binary vector
        global_vec = torch.tensor(
            [(self.phase >> 1) & 1, self.phase & 1], dtype=torch.float32
        )

        return (x, ei, global_vec.unsqueeze(0)), {
            "observation": init_observation.observation
        }

    def step(self, action: Union[Action, int]):
        if isinstance(action, int):
            action = self.graph.int_to_action(action)
        action = self.graph.add_flood_to_action_list(action)
        result = self.original_env.step_parallel({self.agent_name: action})[
            self.agent_name
        ]
        self.graph.parse_observation(action, copy.deepcopy(result.observation))

        # Get graph data/update the state of the graph
        resp = self.graph.get_state()
        x, ei = resp

        # Computer reward, success if OpServer is impacted
        reward = 0.0
        success = False
        for i in range(x.size(0)):
            if (
                isinstance(action, Impact)
                and "Op_Server" in action.hostname
                and x[i, self.graph.ID_SIZE + HostType.OPSERVER.value] == 1
                and x[
                    i,
                    self.graph.ID_SIZE
                    + HostType.__len__()
                    + CompromiseLevel.Impacted.value,
                ]
                == 1
            ):
                reward += 1.0
                success = True

        if self.bt_config is not None:
            bt_reward, self.action_mask, self.phase = self.bt.step_reward(
                state=(x.numpy(), ei.numpy()), mission_str=self.mission_str
            )
            reward += bt_reward
        global_vec = torch.tensor(
            [(self.phase >> 1) & 1, self.phase & 1], dtype=torch.float32
        )

        info = {
            "observation": result.observation,
            "action_space": result.action_space,
            "success": success,
        }
        return (x, ei, global_vec.unsqueeze(0)), reward, success, False, info
