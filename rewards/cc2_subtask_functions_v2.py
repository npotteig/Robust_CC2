import numpy as np
from typing import Tuple

from CybORG.Shared.Actions import Action

from wrapper.red_observation_graph import HostType, CompromiseLevel, RedObservationGraph


def is_discovered_function(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, node_id: int, action: Action
) -> bool:
    x, ei = state
    idx = RedObservationGraph.ID_SIZE + HostType.__len__()
    return x[node_id, idx : idx + CompromiseLevel.__len__()].sum() > 0


def is_compromised_function(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, node_id: int, action: Action
) -> bool:
    x, ei = state
    idx = (
        RedObservationGraph.ID_SIZE
        + HostType.__len__()
        + CompromiseLevel.PrivEscalated.value
    )
    return x[node_id, idx : idx + 2].sum() > 0


def is_routable_function_subnet1(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, node_id: int, action: Action
) -> bool:
    x, ei = state
    if len(ei) == 0:
        return False
    src = ei[0, :]

    user = RedObservationGraph.ID_SIZE + HostType.USER.value
    privesc = (
        RedObservationGraph.ID_SIZE
        + HostType.__len__()
        + CompromiseLevel.PrivEscalated.value
    )

    is_user = x[:, user] == 1
    is_privesc = x[:, privesc] == 1

    valid_edges = (
        is_user[src]  # src is user
        & is_privesc[src]  # src has privilege escalation or is impacted
    )

    return np.any(valid_edges)


def is_routable_function_subnet2(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, node_id: int, action: Action
) -> bool:
    x, ei = state
    if len(ei) == 0:
        return False
    src = ei[0, :]

    user = RedObservationGraph.ID_SIZE + HostType.USER.value
    enterprise = RedObservationGraph.ID_SIZE + HostType.ENTERPISE.value
    privesc = (
        RedObservationGraph.ID_SIZE
        + HostType.__len__()
        + CompromiseLevel.PrivEscalated.value
    )

    is_user = x[:, user] == 1
    is_enterprise = x[:, enterprise] == 1
    is_privesc = x[:, privesc] == 1

    valid_edges = is_privesc[src] & (
        is_user[src]  # src is user
        | is_enterprise[src]  # src is enterprise
    )

    return np.any(valid_edges)


def is_routable_function_subnet3(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, node_id: int, action: Action
) -> bool:
    x, ei = state
    if len(ei) == 0:
        return False
    src = ei[0, :]

    enterprise = RedObservationGraph.ID_SIZE + HostType.ENTERPISE.value
    opserver = RedObservationGraph.ID_SIZE + HostType.OPSERVER.value
    privesc = (
        RedObservationGraph.ID_SIZE
        + HostType.__len__()
        + CompromiseLevel.PrivEscalated.value
    )
    impacted = (
        RedObservationGraph.ID_SIZE
        + HostType.__len__()
        + CompromiseLevel.Impacted.value
    )

    is_enterprise = x[:, enterprise] == 1
    is_opserver = x[:, opserver] == 1
    is_privesc = x[:, privesc] == 1
    is_impacted = x[:, impacted] == 1

    valid_edges = (is_privesc[src] | is_impacted[src]) & (
        is_enterprise[src]  # src is enterprise
        | is_opserver[src]  # src is op server
    )

    return np.any(valid_edges)
