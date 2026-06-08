import math
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import enum
from ipaddress import IPv4Address, IPv4Network

import torch
import numpy as np

from CybORG.Shared.Actions import (
    PrivilegeEscalate,
    ExploitRemoteService,
    Impact,
    DiscoverNetworkServices,
    Sleep,
    DiscoverRemoteSystems,
    FloodPort,
)
from CybORG.Shared.Enums import TrinaryEnum
from CybORG.Shared.Actions import Action


class HostType(enum.Enum):
    UNKNOWN = 0
    USER = 1
    ENTERPISE = 2
    OPHOST = 3
    OPSERVER = 4


class CompromiseLevel(enum.Enum):
    RemoteDiscovered = 0
    NetServicesDiscovered = 1
    Exploited = 2
    PrivEscalated = 3
    Impacted = 4  # Only used for OpServer


@dataclass
class RedNode:
    id: int
    hostname: Optional[str] = None
    host_type: Optional[HostType] = HostType.UNKNOWN
    subnet: Optional[IPv4Network] = None
    ip_address: Optional[IPv4Address] = None
    compromise_level: Optional[CompromiseLevel] = CompromiseLevel.RemoteDiscovered


def pad_observation(x, max_nodes):
    assert x.shape[0] <= max_nodes
    if x.shape[0] < max_nodes:
        padding = torch.zeros((max_nodes - x.shape[0], x.shape[1]), dtype=torch.float)
        x = torch.cat([x, padding], dim=0)
    return x


class RedObservationGraph:
    ID_SIZE = 4

    NUM_NODE_ACTIONS = 2  # DiscoverRemoteSystems, Attack
    INVALID_KEYS = ["success", "Active_Sessions"]
    DIM = (
        ID_SIZE + HostType.__len__() + CompromiseLevel.__len__() + 1
    )  # one hot host type, one hot compromise level, degree

    def __init__(
        self, max_nodes=None, add_flood_actions=False, add_stealth_action=False
    ):
        self.session = 0
        self.agent_name = "Red"
        self.max_nodes = max_nodes
        self.add_flood_actions = add_flood_actions
        self.add_stealth_action = add_stealth_action
        self.stealth_detection_rate = 0.1 if add_stealth_action else 0.95

        self.__class__.ID_SIZE = (
            max_nodes if max_nodes is not None else 13
        ).bit_length()
        # self.__class__.ID_SIZE = 0
        self.__class__.DIM = (
            self.ID_SIZE + HostType.__len__() + CompromiseLevel.__len__() + 1
        )
        if self.add_flood_actions:
            self.NUM_NODE_ACTIONS = (
                4  # DiscoverRemoteSystems, Attack/StealthAttack, Flood, Stop Flood
            )
            self.DIM = (
                self.ID_SIZE + HostType.__len__() + CompromiseLevel.__len__() + 1 + 1
            )  # is_flooded
        else:
            self.NUM_NODE_ACTIONS = 2  # DiscoverRemoteSystems, Attack/StealthAttack
        self.flooded_nodes = set()
        
    def setup(self, initial_observation: dict):
        self.nodes: Dict[int, RedNode] = dict()  # id: RedNode
        self.host_map: Dict[str, RedNode] = dict()
        self.ip_map: Dict[IPv4Address, RedNode] = dict()
        self.nid = 0
        self.active_sessions = initial_observation["Active_Sessions"]
        self.edges = set()
        self.flooded_nodes = set()

        # User 0
        initial_observation.pop("success")
        hostname = list(initial_observation)[0]
        ip_address = initial_observation[hostname]["Interface"][0]["IP Address"]
        subnet = initial_observation[hostname]["Interface"][0]["Subnet"]
        init_node = RedNode(
            id=self.nid,
            hostname=hostname,
            host_type=HostType.USER,
            subnet=subnet,
            ip_address=ip_address,
            compromise_level=CompromiseLevel.PrivEscalated,
        )
        self.ip_map[ip_address] = init_node
        self.host_map[hostname] = init_node
        self.nodes[self.nid] = init_node

        self.nid += 1

    def parse_observation(self, action: Action, observation: dict):
        success = observation.pop("success")
        if isinstance(action, DiscoverRemoteSystems) and success == TrinaryEnum.TRUE:
            subnet = action.subnet
            prev_nid = self.nid
            for key, value in observation.items():
                if key in self.INVALID_KEYS:
                    continue
                ip_address = value["Interface"][0]["IP Address"]
                if ip_address not in self.ip_map:
                    new_node = RedNode(
                        id=self.nid,
                        subnet=subnet,
                        ip_address=ip_address,
                    )
                    self.nodes[self.nid] = new_node
                    self.ip_map[ip_address] = new_node
                    self.nid += 1
            if self.nid > prev_nid:
                # Find nodes that are connected to the subnet or in the subnet
                in_subnet = [
                    node.id for node in self.nodes.values() if node.subnet == subnet
                ]
                # connected_to_subnet = [src_nid for src_nid, dst_nid in self.edges if dst_nid in in_subnet]
                # all_connected = set(in_subnet + connected_to_subnet)
                # Add edges between all nodes in subnet and all nodes connected to subnet
                for src_nid in in_subnet:
                    for dst_nid in in_subnet:
                        if src_nid != dst_nid:
                            self.edges.add((src_nid, dst_nid))
        elif (
            isinstance(action, DiscoverNetworkServices) and success == TrinaryEnum.TRUE
        ):
            ip_address = action.ip_address
            node = self.ip_map.get(ip_address)
            if node is not None:
                node.compromise_level = CompromiseLevel.NetServicesDiscovered
        elif isinstance(action, ExploitRemoteService) and success == TrinaryEnum.TRUE:
            ip_address = action.ip_address
            node = self.ip_map.get(ip_address)
            if node is not None:
                node.compromise_level = CompromiseLevel.Exploited
                node.hostname = observation[str(ip_address)]["System info"]["Hostname"]
                if "Enterprise" in node.hostname:
                    node.host_type = HostType.ENTERPISE
                elif "Op_Server" in node.hostname:
                    node.host_type = HostType.OPSERVER
                elif "Op_Host" in node.hostname:
                    node.host_type = HostType.OPHOST
                else:
                    node.host_type = HostType.USER
                self.host_map[node.hostname] = node
        elif isinstance(action, PrivilegeEscalate) and success == TrinaryEnum.TRUE:
            hostname = action.hostname
            if hostname in observation:
                node = self.host_map.get(hostname)
                # Good if subnet is not already known in the case of moving from User to Enterprise
                subnet = observation[hostname]["Interface"][0]["Subnet"]
                if node is not None:
                    node.compromise_level = CompromiseLevel.PrivEscalated
                    node.subnet = subnet
                for key, value in observation.items():
                    if key in self.INVALID_KEYS or key == hostname:
                        continue
                    ip_address = value["Interface"][0]["IP Address"]
                    if ip_address not in self.ip_map:
                        host_type = HostType.USER
                        if "Enterprise" in key:
                            host_type = HostType.ENTERPISE
                        elif "Op_Server" in key:
                            host_type = HostType.OPSERVER
                        elif "Op_Host" in key:
                            host_type = HostType.OPHOST
                        new_node = RedNode(
                            id=self.nid,
                            hostname=key,
                            host_type=host_type,
                            ip_address=ip_address,
                        )
                        self.nodes[self.nid] = new_node
                        self.ip_map[ip_address] = new_node
                        self.nid += 1
                        self.edges.add((node.id, new_node.id))
                        self.edges.add((new_node.id, node.id))
                        # Add edge for every node in node subnet
                        # for src_nid, other_node in self.nodes.items():
                        #     if other_node.subnet == subnet:
                        #         self.edges.add((other_node.id, new_node.id))
                        #         self.edges.add((new_node.id, other_node.id))

        elif isinstance(action, Impact) and success == TrinaryEnum.TRUE:
            hostname = action.hostname
            node = self.host_map.get(hostname)
            if node is not None:
                node.compromise_level = CompromiseLevel.Impacted

        # De-escalate compromise level based on change in active sessions
        active_sessions: dict = observation.pop("Active_Sessions")
        # Handle removed sessions
        if len(active_sessions) <= len(self.active_sessions):
            removed_hosts = self.active_sessions.keys() - active_sessions.keys()
            for hostname in removed_hosts:
                node = self.host_map.get(hostname)
                if (
                    node
                    and node.compromise_level.value
                    > CompromiseLevel.NetServicesDiscovered.value
                ):
                    node.compromise_level = CompromiseLevel.NetServicesDiscovered

        # Build new session dict
        new_session = {}
        for hostname, session in active_sessions.items():
            if hostname in self.active_sessions and self.active_sessions[hostname][
                "Username"
            ] in ("root", "SYSTEM"):
                session = dict(session)  # avoid mutating original
                session["Username"] = self.active_sessions[hostname]["Username"]
            new_session[hostname] = session

        self.active_sessions = new_session

    def get_state(self, include_names=False):
        # Convert to PyG edge index
        if len(self.edges) > 0:
            ei = torch.tensor(list(self.edges), dtype=torch.long).t().contiguous()
        else:
            ei = torch.empty((2, 0), dtype=torch.long)

        x = torch.zeros((len(self.nodes), self.DIM - self.ID_SIZE), dtype=torch.float)
        names = [None] * len(self.nodes)
        for nid, node in self.nodes.items():
            x[
                nid,
                [
                    node.host_type.value,
                    HostType.__len__() + node.compromise_level.value,
                    -1,
                ],
            ] = 1.0
            if self.add_flood_actions and nid in self.flooded_nodes:
                x[nid, -2] = 1.0
            names[nid] = node.hostname

        if self.max_nodes is not None:
            x = pad_observation(x, self.max_nodes)
        ids = torch.arange(self.max_nodes if self.max_nodes is not None else 13)
        binary_ids = (
            (ids.unsqueeze(1) >> torch.arange(self.ID_SIZE - 1, -1, -1)) & 1
        ).float()

        x = torch.cat([binary_ids, x], dim=1)

        if not include_names:
            return x, ei
        return x, ei, names

    def action_to_int(self, action: Action):
        offset = 1
        if isinstance(action, Sleep):
            return 0
        elif isinstance(action, DiscoverRemoteSystems):
            subnet = action.subnet
            for node in self.nodes.values():
                if (
                    node.subnet == subnet
                    and node.compromise_level.value
                    >= CompromiseLevel.PrivEscalated.value
                ):
                    return node.id * self.NUM_NODE_ACTIONS + offset
        elif isinstance(action, (DiscoverNetworkServices, ExploitRemoteService)):
            node = self.ip_map.get(action.ip_address)
            if node is not None:
                return node.id * self.NUM_NODE_ACTIONS + 1 + offset
        elif isinstance(action, (PrivilegeEscalate, Impact)):
            node = self.host_map.get(action.hostname)
            if node is not None:
                return node.id * self.NUM_NODE_ACTIONS + 1 + offset
        return None

    def int_to_action(self, action_int: int):
        if action_int == 0:
            return Sleep()
        node_id = (action_int - 1) // self.NUM_NODE_ACTIONS
        action_type = (action_int - 1) % self.NUM_NODE_ACTIONS
        node = self.nodes.get(node_id, None)
        if action_type == 0 and node_id not in self.flooded_nodes:
            if (
                node is not None
                and node.subnet is not None
                and node.compromise_level.value >= CompromiseLevel.PrivEscalated.value
            ):
                return DiscoverRemoteSystems(
                    session=self.session, agent=self.agent_name, subnet=node.subnet
                )
        elif action_type == 1 and node_id not in self.flooded_nodes:
            if node is not None:
                if (
                    node.compromise_level.value
                    == CompromiseLevel.RemoteDiscovered.value
                ):
                    return DiscoverNetworkServices(
                        session=self.session,
                        agent=self.agent_name,
                        ip_address=node.ip_address,
                    )
                elif (
                    node.compromise_level.value
                    == CompromiseLevel.NetServicesDiscovered.value
                ):
                    return ExploitRemoteService(
                        session=self.session,
                        agent=self.agent_name,
                        ip_address=node.ip_address,
                        detection_rate=self.stealth_detection_rate,
                    )
                elif node.compromise_level.value == CompromiseLevel.Exploited.value:
                    return PrivilegeEscalate(
                        session=self.session,
                        agent=self.agent_name,
                        hostname=node.hostname,
                    )
                elif (
                    node.compromise_level.value == CompromiseLevel.PrivEscalated.value
                    or node.compromise_level.value == CompromiseLevel.Impacted.value
                ):
                    if node.host_type == HostType.OPSERVER:
                        return Impact(
                            session=self.session,
                            agent=self.agent_name,
                            hostname=node.hostname,
                        )
                    else:
                        return PrivilegeEscalate(
                            session=self.session,
                            agent=self.agent_name,
                            hostname=node.hostname,
                        )
        elif self.add_flood_actions and action_type == 2:
            if (
                node is not None
                and node.compromise_level.value
                >= CompromiseLevel.NetServicesDiscovered.value
            ):
                self.flooded_nodes.add(node_id)
            return Sleep()
        elif self.add_flood_actions and action_type == 3:
            self.flooded_nodes.discard(node_id)
            return Sleep()

        return Sleep()

    def add_flood_to_action_list(self, action: Action):
        actions = [action]
        for flood_action in self.flooded_nodes:
            flood_node = self.nodes.get(flood_action, None)
            if flood_node is not None:
                actions.insert(
                    0,
                    FloodPort(
                        session=self.session,
                        agent=self.agent_name,
                        ip_address=flood_node.ip_address,
                    ),
                )
        return actions


def red_goal_1_function(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, node_id: int, action: Action
):
    x, _ = state
    compromised_ratio = math.ceil(
        0.75 * x.shape[0]
    )  # 10 if this is 13 nodes (excludes defender)
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
    return (
        (x[:, privesc] == 1).sum() + (x[:, impacted] == 1).sum()
    ) >= compromised_ratio


def red_goal_2_function(
    state: Tuple[np.ndarray, np.ndarray], mission_str: str, node_id: int, action: Action
):
    x, _ = state
    for i in range(x.shape[0]):
        if (
            isinstance(action, Impact)
            and "Op_Server0" in action.hostname
            and x[i, RedObservationGraph.ID_SIZE + HostType.OPSERVER.value] == 1
            and x[i, -2] == 1
        ):
            return True
    return False
