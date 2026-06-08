import py_trees as pt
from typing import Callable, Tuple
import numpy as np

from CybORG.Shared.Actions.Action import Action


class SubtaskCompleted(pt.behaviour.Behaviour):
    """
    A behaviour that checks if a subtask has been completed.
    """

    def __init__(
        self,
        name: str,
        subtask_function: Callable[[Tuple[np.ndarray, np.ndarray], str], bool],
        use_z3: bool = False,
    ):
        super().__init__(name=name)
        self.subtask_function = subtask_function
        self.use_z3 = use_z3

        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key(key="state", access=pt.common.Access.READ)
        self.blackboard.register_key(key="mission_str", access=pt.common.Access.READ)
        self.blackboard.register_key(key="reward", access=pt.common.Access.WRITE)

    def setup(self, **kwargs):
        self.memory_status = pt.common.Status.FAILURE

    def update(self):
        res = self.subtask_function(self.blackboard.state, self.blackboard.mission_str)
        if res:
            if self.memory_status == pt.common.Status.FAILURE:
                self.blackboard.reward += 0.5
            self.memory_status = pt.common.Status.SUCCESS
            return pt.common.Status.SUCCESS
        if self.memory_status == pt.common.Status.SUCCESS:
            self.blackboard.reward += -0.5
        self.memory_status = pt.common.Status.FAILURE
        return pt.common.Status.FAILURE


class ConnectPort(pt.behaviour.Behaviour):
    """
    A behaviour that checks if a host has been priviledge escalated and has a port connected to the next subnet (i.e. user to enterprise or enterprise to op server)
    """

    def __init__(
        self,
        name: str,
        subtask_port_function: Callable[[Tuple[np.ndarray, np.ndarray], str], bool],
        action_mask: np.ndarray,
        use_z3: bool = False,
    ):
        super().__init__(name=name)
        self.port_function = subtask_port_function
        self.action_mask = action_mask
        self.use_z3 = use_z3

        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key(key="state", access=pt.common.Access.READ)
        self.blackboard.register_key(key="mission_str", access=pt.common.Access.READ)
        self.blackboard.register_key(key="reward", access=pt.common.Access.WRITE)
        self.blackboard.register_key(key="action_mask", access=pt.common.Access.WRITE)

    def setup(self, **kwargs):
        self.memory_status = pt.common.Status.RUNNING

    def update(self):
        res = self.port_function(self.blackboard.state, self.blackboard.mission_str)
        if res:
            if self.memory_status == pt.common.Status.RUNNING:
                self.blackboard.reward += 0.5
            self.memory_status = pt.common.Status.SUCCESS
            return pt.common.Status.SUCCESS
        if self.memory_status == pt.common.Status.SUCCESS:
            self.blackboard.reward += -0.5
        self.blackboard.action_mask = self.action_mask
        self.memory_status = pt.common.Status.RUNNING
        return pt.common.Status.RUNNING


class ExploitSubnet(pt.behaviour.Behaviour):
    """
    A behaviour that checks if the agent has exploited a host on the next subnet
    """

    def __init__(self, name: str, action_mask: np.ndarray):
        super().__init__(name=name)

        self.action_mask = action_mask
        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key(key="action_mask", access=pt.common.Access.WRITE)

    def update(self):
        self.blackboard.action_mask = self.action_mask
        return pt.common.Status.RUNNING


class RewardBehaviour(pt.behaviour.Behaviour):
    """
    A behaviour that checks if the agent has escalated privileges on a host
    """

    def __init__(
        self,
        name: str,
        subtask_function: Callable[
            [Tuple[np.ndarray, np.ndarray], str, Action, int | None], bool
        ],
        action_mask: np.ndarray,
        node_id: int | None = None,
        condition_behaviour: bool = False,
        r_1: float = 0.5,
        r_2: float = 0.0,
        phase: int | None = None,
    ):
        super().__init__(name=name)
        self.subtask_function = subtask_function
        self.action_mask = action_mask
        self.node_id = node_id
        self.r_1 = r_1
        self.r_2 = r_2
        self.phase = phase
        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key(key="state", access=pt.common.Access.READ)
        self.blackboard.register_key(key="mission_str", access=pt.common.Access.READ)
        self.blackboard.register_key(key="action", access=pt.common.Access.READ)
        self.blackboard.register_key(key="reward", access=pt.common.Access.WRITE)
        self.blackboard.register_key(key="action_mask", access=pt.common.Access.WRITE)
        self.blackboard.register_key(key="phase", access=pt.common.Access.WRITE)
        self.unsuccessful_status = (
            pt.common.Status.FAILURE
            if condition_behaviour
            else pt.common.Status.RUNNING
        )

    def setup(self, **kwargs):
        self.memory_status = self.unsuccessful_status

    def update(self):
        if self.node_id is None:
            res = self.subtask_function(
                self.blackboard.state, self.blackboard.mission_str, self.blackboard.action
            )
        else:
            res = self.subtask_function(
                self.blackboard.state, self.blackboard.mission_str, self.blackboard.action, self.node_id
            )
        if res:
            if self.memory_status == self.unsuccessful_status:
                self.blackboard.reward += self.r_1
            self.blackboard.reward += self.r_2
            self.memory_status = pt.common.Status.SUCCESS
            self.blackboard.phase = (
                self.phase if self.phase is not None else self.blackboard.phase
            )
            return pt.common.Status.SUCCESS
        if self.memory_status == pt.common.Status.SUCCESS:
            self.blackboard.reward += -self.r_1
        self.memory_status = self.unsuccessful_status
        self.blackboard.action_mask = self.action_mask
        self.blackboard.phase = (
            self.phase if self.phase is not None else self.blackboard.phase
        )
        return self.unsuccessful_status


class RunningBehaviour(pt.behaviour.Behaviour):
    """
    A behaviour that always returns running
    """

    def __init__(self, name: str, action_mask: np.ndarray, phase: int | None = None):
        super().__init__(name=name)

        self.action_mask = action_mask
        self.phase = phase
        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key(key="action_mask", access=pt.common.Access.WRITE)
        self.blackboard.register_key(key="phase", access=pt.common.Access.WRITE)

    def update(self):
        self.blackboard.action_mask = self.action_mask
        self.blackboard.phase = (
            self.phase if self.phase is not None else self.blackboard.phase
        )
        return pt.common.Status.RUNNING
