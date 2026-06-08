import numpy as np
from typing import Tuple

from CybORG.Shared.Actions import Action, Impact
from wrapper.red_observation_graph import HostType, CompromiseLevel, RedObservationGraph


def impact_complete(state: Tuple[np.ndarray, np.ndarray], mission_str: str, action: Action) -> bool:
    x, ei = state
    for i in range(x.shape[0]):
        if (
            isinstance(action, Impact)
            and "Op_Server" in action.hostname
            and x[i, RedObservationGraph.ID_SIZE + HostType.OPSERVER.value] == 1
            and x[
                i,
                RedObservationGraph.ID_SIZE
                + HostType.__len__()
                + CompromiseLevel.Impacted.value,
            ]
            == 1
        ):
            return True
    return False

def subtask_1_complete(state: Tuple[np.ndarray, np.ndarray], mission_str: str, _) -> bool:
    # Check if agent has presence on enterprise subnet or presence on op server
    x, ei = state
    ent_escalated, op_exploited = False, False
    exploited = (
        RedObservationGraph.ID_SIZE
        + HostType.__len__()
        + CompromiseLevel.Exploited.value
    )
    for node in x:
        if node[RedObservationGraph.ID_SIZE + HostType.ENTERPISE.value] == 1 and np.any(
            node[exploited : exploited + 3] == 1
        ):
            ent_escalated = True
        if (
            node[RedObservationGraph.ID_SIZE + HostType.OPSERVER.value] == 1
            or node[RedObservationGraph.ID_SIZE + HostType.OPHOST.value] == 1
        ) and np.any(node[exploited : exploited + 3] == 1):
            op_exploited = True
    return ent_escalated or op_exploited


def subtask_1_port_connected(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, _
) -> bool:
    # Check if red has port connected from user subnet to enterprise subnet with privesc on user host
    x, ei = state
    if len(ei) == 0:
        return False
    # src = ei[0, :]
    # dst = ei[1, :]

    enterprise = RedObservationGraph.ID_SIZE + HostType.ENTERPISE.value
    user = RedObservationGraph.ID_SIZE + HostType.USER.value
    privesc = (
        RedObservationGraph.ID_SIZE
        + HostType.__len__()
        + CompromiseLevel.PrivEscalated.value
    )

    # Boolean masks for node conditions
    is_enterprise = x[:, enterprise] == 1
    is_user = x[:, user] == 1
    is_privesc = x[:, privesc] == 1

    # Apply conditions to edges
    valid_edges = (
        is_user[:]  # src is user
        & is_privesc[:]  # src has privilege escalation
        # is_enterprise[dst]        # dst is enterprise
    )

    return np.any(valid_edges) and np.any(is_enterprise[:])


def subtask_2_complete(state: Tuple[np.ndarray, np.ndarray], mission_str: str, _) -> bool:
    # Check if agent has session on OpServer
    x, ei = state
    exploited = (
        RedObservationGraph.ID_SIZE
        + HostType.__len__()
        + CompromiseLevel.Exploited.value
    )
    for node in x:
        if node[RedObservationGraph.ID_SIZE + HostType.OPSERVER.value] == 1 and np.any(
            node[exploited : exploited + 3] == 1
        ):
            return True
    return False


def subtask_2_port_connected(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, _
) -> bool:
    # Check if red has port connected from enterprise subnet to op server with privesc on enterprise host
    x, ei = state
    if len(ei) == 0:
        return False
    # src = ei[0, :]
    # dst = ei[1, :]

    op_server = RedObservationGraph.ID_SIZE + HostType.OPSERVER.value
    op_host = RedObservationGraph.ID_SIZE + HostType.OPHOST.value
    enterprise = RedObservationGraph.ID_SIZE + HostType.ENTERPISE.value
    privesc = (
        RedObservationGraph.ID_SIZE
        + HostType.__len__()
        + CompromiseLevel.PrivEscalated.value
    )

    is_op_server = x[:, op_server] == 1
    is_op_host = x[:, op_host] == 1
    is_enterprise = x[:, enterprise] == 1
    is_privesc = x[:, privesc] == 1

    valid_edges = (
        is_enterprise[:]  # src is enterprise
        & is_privesc[:]  # src has privilege escalation
        # (is_op_server[dst] | is_op_host[dst])  # dst is op server or op host
    )
    return np.any(valid_edges) and (np.any(is_op_server[:]) or np.any(is_op_host[:]))


def flood_check(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, _, node_int: int
) -> bool:
    x, ei = state
    return x[node_int, -2] == 1
