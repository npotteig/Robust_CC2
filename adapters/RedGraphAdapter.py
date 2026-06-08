import copy
from typing import Optional, Dict, List

import torch

from wrapper.red_observation_graph import CompromiseLevel, RedObservationGraph, HostType
from rewards.bt import BehaviourTreeReward, BehaviourTreeConfig

from CybORG import CybORG
from CybORG.Shared.Results import Results
from CybORG.Shared.Actions import Action, Impact, Sleep

from adapters.BaseAdapter import Adapter


class RedGraphAdapter(Adapter):
    def __init__(
        self,
        env: CybORG,
        max_nodes=None,
        reward_mode: str = "environment",
        bt_config: Optional[BehaviourTreeConfig] = None,
        action_mask_dict: Optional[Dict[str, List[int]]] = None,
        add_flood_actions: bool = False,
        add_stealth_action: bool = False,
        use_bt_phase=True,
    ):
        self.agent_name = "Red"
        self.original_env = env
        self.graph = RedObservationGraph(
            max_nodes=max_nodes,
            add_flood_actions=add_flood_actions,
            add_stealth_action=add_stealth_action,
        )
        self.reward_mode = reward_mode
        self.bt_config = bt_config
        self.action_mask_dict = action_mask_dict
        self.reward_mode = reward_mode
        self.action_mask_dict = action_mask_dict
        self.use_bt_phase = use_bt_phase
        self.global_vec_bits = 2

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

    def pre_reset(self):
        self.sum_flood = 0
        return None

    def post_reset(self, result: Results):
        self.graph = RedObservationGraph(
            max_nodes=self.graph.max_nodes,
            add_flood_actions=self.graph.add_flood_actions,
            add_stealth_action=self.graph.add_stealth_action,
        )
        self.graph.setup(copy.deepcopy(result.observation))
        x, ei = self.graph.get_state()

        self.mission_str = "Impact OpServer"
        self.phase = 0
        if self.bt_config is not None:
            self.bt = BehaviourTreeReward.create_bt(
                mission_str=self.mission_str,
                bt_config=self.bt_config,
                action_masks=self.action_mask_dict,
            )
            _, self.action_mask, bt_phase = self.bt.step_reward(
                state=(x.numpy(), ei.numpy()), mission_str=self.mission_str, action=Sleep()
            )
            if self.use_bt_phase:
                self.phase = bt_phase
        # convert phase to binary vector
        global_vec = torch.tensor(
            [(self.phase >> i) & 1 for i in reversed(range(self.global_vec_bits))],
            dtype=torch.float32
        )

        return (x, ei, global_vec.unsqueeze(0)), {
            "observation": result.observation,
            "action_space": result.action_space,
        }

    def pre_step(self, action=None):
        if isinstance(action, int):
            return self.graph.add_flood_to_action_list(self.graph.int_to_action(action))
        return self.graph.add_flood_to_action_list(action)

    def post_step(self, action: Action, result: Results):
        # if isinstance(action, int):
        #     action = self.graph.int_to_action(action)
        self.graph.parse_observation(action, copy.deepcopy(result.observation))
        # success = observation.observation.get('success', 'UNK')

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
                # reward += 1.0
                success = True
        if self.bt_config is not None:
            bt_reward, self.action_mask, bt_phase = self.bt.step_reward(
                state=(x.numpy(), ei.numpy()), mission_str=self.mission_str, action=action
            )
            reward += bt_reward
            if self.use_bt_phase:
                self.phase = bt_phase
        else:
            reward += result.reward
        global_vec = torch.tensor(
            [(self.phase >> i) & 1 for i in reversed(range(self.global_vec_bits))],
            dtype=torch.float32
        )
        info = {
            "observation": result.observation,
            "action_space": result.action_space,
            "success": success,
        }
        return (x, ei, global_vec.unsqueeze(0)), reward, success, False, info
