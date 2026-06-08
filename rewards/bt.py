from typing import Sequence, Callable, Tuple, Dict, Optional, List
import numpy as np
import py_trees as pt
from pydantic import BaseModel

from CybORG.Shared.Actions import Action

from rewards.behaviours import (
    RewardBehaviour,
    RunningBehaviour,
)

from rewards.parallel_v2 import ParallelV2, ParallelPolicy


class BehaviourTreeConfig(BaseModel):
    subtask_names: Sequence[str]
    subtask_complete_functions: Sequence[
        Callable[[Tuple[np.ndarray, np.ndarray], str, Action], bool]
    ]
    subtask_port_functions: Sequence[
        Callable[[Tuple[np.ndarray, np.ndarray], str, Action], bool]
    ]
    subtask_flood_function: Optional[
        Callable[[Tuple[np.ndarray, np.ndarray], str, Action, int], bool]
    ] = None
    mission_complete_function: Optional[Callable[[Tuple[np.ndarray, np.ndarray], str, Action], bool]] = None
    flood_k: Optional[int] = None
    flood_node_ids: Optional[List[int]] = None
    use_memory: bool = False


class BehaviourTreeReward:
    def __init__(self, root: pt.behaviour.Behaviour, blackboard: pt.blackboard.Client):
        self.root = root
        self.root.setup_with_descendants()
        self.blackboard = blackboard

    def step_reward(
        self, state: np.ndarray, mission_str: str, action: Action
    ) -> Tuple[float, np.ndarray]:
        """
        Step the behaviour tree and return the reward based on the updates of the behaviour nodes.
        """
        self.blackboard.set("state", state)
        self.blackboard.set("mission_str", mission_str)
        self.blackboard.set("action", action)
        self.blackboard.set("reward", 0.0)
        self.blackboard.set("action_mask", None)
        # self.blackboard.set("phase", 0)
        self.root.tick_once()
        return (
            self.blackboard.reward,
            self.blackboard.action_mask,
            self.blackboard.phase,
        )

    @classmethod
    def _create_subtask(
        cls,
        subtask_name: str,
        subtask_complete_function: Callable[[Tuple[np.ndarray, np.ndarray], str, Action], bool],
        subtask_port_function: Callable[[Tuple[np.ndarray, np.ndarray], str, Action], bool],
        use_memory: bool = False,
        action_masks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        phase: Optional[int] = None,
    ):
        """
        Create a subtask behaviour that checks if the subtask is completed (escalated to next subnet)
        and if the credentials have been discovered to escalate to the next subnet.
        """
        return pt.composites.Selector(
            name=subtask_name,
            children=[
                RewardBehaviour(
                    name=f"{subtask_name}_completed",
                    subtask_function=subtask_complete_function,
                    condition_behaviour=True,
                    action_mask=None,
                    phase=phase,
                ),
                pt.composites.Sequence(
                    name=f"Sequence {subtask_name}",
                    children=[
                        RewardBehaviour(
                            name=f"{subtask_name}_connect_port",
                            subtask_function=subtask_port_function,
                            action_mask=action_masks[0] if action_masks else None,
                            phase=phase,
                        ),
                        RunningBehaviour(
                            name=f"{subtask_name}_exploit_subnet",
                            action_mask=action_masks[1] if action_masks else None,
                            phase=phase,
                        ),
                    ],
                    memory=use_memory,
                ),
            ],
            memory=False,
        )

    @classmethod
    def _create_parallel(
        cls,
        name: str,
        flood_k: int,
        flood_node_ids: List[int],
        subtask_function: Callable[[Tuple[np.ndarray, np.ndarray], str, Action, int], bool],
        phase: Optional[int] = None,
    ):
        """
        Create a parallel behaviour that checks if the agent has exploited to at least k hosts in the next subnet.
        """
        return ParallelV2(
            name=name,
            policy=ParallelPolicy.SuccessOnK(k=flood_k, synchronise=False),
            children=[
                RewardBehaviour(
                    name=f"Flood_Host_{node_id}",
                    subtask_function=subtask_function,
                    node_id=node_id,
                    r_1=0.1,
                    action_mask=None,
                    phase=phase,
                )
                for node_id in flood_node_ids
            ],
        )

    @classmethod
    def create_bt(
        cls,
        mission_str: str,
        bt_config: BehaviourTreeConfig,
        action_masks: Optional[Dict[str, List[int]]] = None,
    ):
        assert (
            len(bt_config.subtask_names)
            == len(bt_config.subtask_complete_functions)
            == len(bt_config.subtask_port_functions)
        ), "Subtask names, functions, and port functions must have the same length."

        blackboard = pt.blackboard.Client(name="Blackboard")
        blackboard.register_key(key="state", access=pt.common.Access.WRITE)
        blackboard.register_key(key="mission_str", access=pt.common.Access.WRITE)
        blackboard.register_key(key="action", access=pt.common.Access.WRITE)
        blackboard.register_key(key="reward", access=pt.common.Access.WRITE)
        blackboard.register_key(key="action_mask", access=pt.common.Access.WRITE)
        blackboard.register_key(key="phase", access=pt.common.Access.WRITE)
        blackboard.set("mission_str", mission_str)
        blackboard.set("reward", 0.0)
        blackboard.set("phase", 0)
        seq_root = pt.composites.Sequence(name="SeqRoot", memory=bt_config.use_memory)
        for i, (
            subtask_name,
            subtask_complete_function,
            subtask_port_function,
        ) in enumerate(
            zip(
                bt_config.subtask_names,
                bt_config.subtask_complete_functions,
                bt_config.subtask_port_functions,
            )
        ):
            subtask_child = cls._create_subtask(
                subtask_name=subtask_name,
                subtask_complete_function=subtask_complete_function,
                subtask_port_function=subtask_port_function,
                use_memory=bt_config.use_memory,
                action_masks=(
                    np.array(
                        action_masks.get(f"{subtask_name}_1", None),
                        dtype=np.int32,
                    ),
                    np.array(
                        action_masks.get(f"{subtask_name}_2", None),
                        dtype=np.int32,
                    ),
                )
                if action_masks
                else None,
                phase=i+1,
            )
            seq_root.add_child(subtask_child)
        seq_root.add_child(
            RewardBehaviour(
                name="Impact OpServer",
                subtask_function=bt_config.mission_complete_function,
                condition_behaviour=False,
                action_mask=None,
                r_1=1.0,
                r_2=1.0,
                phase=3
            )
        )
        if bt_config.flood_k is None:
            return cls(root=seq_root, blackboard=blackboard)

        parallel_child = cls._create_parallel(
            name="ParallelFlood",
            flood_k=bt_config.flood_k,
            flood_node_ids=bt_config.flood_node_ids,
            subtask_function=bt_config.subtask_flood_function,
            phase=0
        )
        # parallel_root = ParallelV2(
        #     name="ParallelRoot",
        #     policy=ParallelPolicy.SuccessOnAll(synchronise=True),
        #     children=[parallel_child, seq_root]
        # )
        parallel_root = pt.composites.Sequence(
            name="ParallelRoot", children=[parallel_child, seq_root], memory=True
        )
        return cls(root=parallel_root, blackboard=blackboard)
