from typing import Sequence, Callable, Tuple, Optional, List
import numpy as np
import py_trees as pt
from pydantic import BaseModel

from CybORG.Shared.Actions import Action

from rewards.behaviours import RewardBehaviour, RunningBehaviour


class BehaviourTreeConfig(BaseModel):
    subtask_names: Sequence[str]
    node_ids: Sequence[List[int]]
    is_discovered_functions: Sequence[
        Callable[[Tuple[np.ndarray, np.ndarray], str, int, Action], bool]
    ]
    is_routable_functions: Sequence[
        Callable[[Tuple[np.ndarray, np.ndarray], str, int, Action], bool]
    ]
    is_compromised_functions: Sequence[
        Callable[[Tuple[np.ndarray, np.ndarray], str, int, Action], bool]
    ]
    goal_function: Callable[[Tuple[np.ndarray, np.ndarray], str, Action], bool]
    use_memory: bool = False
    is_root_sequence: bool = True


class BehaviourTreeReward:
    def __init__(self, root: pt.behaviour.Behaviour, blackboard: pt.blackboard.Client):
        self.root = root
        self.root.setup_with_descendants()
        self.blackboard = blackboard

    def step_reward(
        self, state: Tuple[np.ndarray, np.ndarray], mission_str: str, action: Action
    ) -> Tuple[float, np.ndarray]:
        """
        Step the behaviour tree and return the reward based on the updates of the behaviour nodes.
        """
        self.blackboard.set("state", state)
        self.blackboard.set("mission_str", mission_str)
        self.blackboard.set("reward", 0.0)
        self.blackboard.set("action_mask", None)
        self.blackboard.set("action", action)
        self.root.tick_once()
        return self.blackboard.reward, self.blackboard.action_mask

    @classmethod
    def _create_node_subtask(
        cls,
        node_id: int,
        is_discovered_function: Callable[
            [Tuple[np.ndarray, np.ndarray], str, int, Action], bool
        ],
        is_routable_function: Callable[
            [Tuple[np.ndarray, np.ndarray], str, int, Action], bool
        ],
        is_compromised_function: Callable[
            [Tuple[np.ndarray, np.ndarray], str, int, Action], bool
        ],
        use_memory: bool = False,
        action_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        """
        Node subtask with condition that checks if node has been compromised,
        if not sequence of discovery, ensuring routability, then attacking to compromise. If use_memory is true, then the reward will only be given the first time the node is compromised.
        """
        return pt.composites.Selector(
            name=f"Node {node_id} Selector",
            children=[
                RewardBehaviour(
                    name=f"Node {node_id} Compromised",
                    subtask_function=is_compromised_function,
                    action_mask=None,
                    node_id=node_id,
                    condition_behaviour=True,
                ),
                pt.composites.Sequence(
                    name=f"Node {node_id} Sequence",
                    children=[
                        RewardBehaviour(
                            name=f"Discover Node {node_id}",
                            subtask_function=is_discovered_function,
                            action_mask=action_masks[0] if action_masks else None,
                            node_id=node_id,
                            condition_behaviour=False,
                        ),
                        RewardBehaviour(
                            name=f"Route to Node {node_id}",
                            subtask_function=is_routable_function,
                            action_mask=action_masks[1] if action_masks else None,
                            node_id=node_id,
                            condition_behaviour=False,
                        ),
                        RunningBehaviour(
                            name=f"Compromise Node {node_id}",
                            action_mask=action_masks[2] if action_masks else None,
                        ),
                    ],
                    memory=use_memory,
                ),
            ],
            memory=False,
        )

    @classmethod
    def _create_parallel_subtask(
        cls,
        subtask_name: str,
        node_ids: List[int],
        is_discovered_function: Callable[
            [Tuple[np.ndarray, np.ndarray], str, int, Action], bool
        ],
        is_routable_function: Callable[
            [Tuple[np.ndarray, np.ndarray], str, int, Action], bool
        ],
        is_compromised_function: Callable[
            [Tuple[np.ndarray, np.ndarray], str, int, Action], bool
        ],
        parallel_policy: pt.common.ParallelPolicy = pt.common.ParallelPolicy.SuccessOnAll(
            synchronise=False
        ),
        use_memory: bool = False,
        action_masks: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ):
        """
        Parallel subtask that checks if all nodes in node_ids have been compromised, if not sequence of discovery, ensuring routability, then attacking to compromise for each node in node_ids. If use_memory is true, then the reward will only be given the first time all nodes are compromised.
        """
        return pt.composites.Parallel(
            name=f"{subtask_name} Parallel",
            children=[
                cls._create_node_subtask(
                    node_id=node_id,
                    is_discovered_function=is_discovered_function,
                    is_routable_function=is_routable_function,
                    is_compromised_function=is_compromised_function,
                    use_memory=use_memory,
                    action_masks=action_masks,
                )
                for node_id in node_ids
            ],
            policy=parallel_policy,
        )

    @classmethod
    def create_bt(
        cls,
        mission_str: str,
        bt_config: BehaviourTreeConfig,
        action_masks: Optional[
            Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ] = None,
    ):
        assert (
            len(bt_config.subtask_names)
            == len(bt_config.node_ids)
            == len(bt_config.is_discovered_functions)
            == len(bt_config.is_routable_functions)
            == len(bt_config.is_compromised_functions)
        ), (
            "Length of subtask_names, node_ids, is_discovered_functions, is_routable_functions, and is_compromised_functions must be the same"
        )
        blackboard = pt.blackboard.Client(name="Blackboard")
        blackboard.register_key(key="state", access=pt.common.Access.WRITE)
        blackboard.register_key(key="mission_str", access=pt.common.Access.WRITE)
        blackboard.register_key(key="reward", access=pt.common.Access.WRITE)
        blackboard.register_key(key="action_mask", access=pt.common.Access.WRITE)
        blackboard.register_key(key="action", access=pt.common.Access.WRITE)
        blackboard.set("mission_str", mission_str)
        blackboard.set("reward", 0.0)
        blackboard.set("action", None)

        children = [
            cls._create_parallel_subtask(
                subtask_name=subtask_name,
                node_ids=node_ids,
                is_discovered_function=is_discovered_function,
                is_routable_function=is_routable_function,
                is_compromised_function=is_compromised_function,
                use_memory=bt_config.use_memory,
                action_masks=action_masks[i] if action_masks else None,
            )
            for i, (
                subtask_name,
                node_ids,
                is_discovered_function,
                is_routable_function,
                is_compromised_function,
            ) in enumerate(
                zip(
                    bt_config.subtask_names,
                    bt_config.node_ids,
                    bt_config.is_discovered_functions,
                    bt_config.is_routable_functions,
                    bt_config.is_compromised_functions,
                )
            )
        ]

        if bt_config.is_root_sequence:
            root = pt.composites.Sequence(
                name="Root", children=children, memory=bt_config.use_memory
            )
        else:
            root = pt.composites.Parallel(
                name="Root",
                children=children,
                policy=pt.common.ParallelPolicy.SuccessOnAll(synchronise=False),
            )

        # Add goal check as a selector to the root
        goal_check = RewardBehaviour(
            name="Goal Check",
            subtask_function=bt_config.goal_function,
            action_mask=None,
            node_id=-1,
            condition_behaviour=True,
        )
        root = pt.composites.Selector(
            name="Root Selector", children=[goal_check, root], memory=False
        )
        return cls(root=root, blackboard=blackboard)
