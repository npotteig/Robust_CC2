from CybORG.Agents import BaseAgent
from CybORG.Shared import Results
from CybORG.Shared.Actions import (
    PrivilegeEscalate,
    ExploitRemoteService,
    DiscoverRemoteSystems,
    Impact,
    DiscoverNetworkServices,
)
from CybORG.Shared.Enums import TrinaryEnum
import random


class B_lineAgent(BaseAgent):
    def __init__(self):
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None
        self.action_history = {}
        self.jumps = [0, 1, 2, 2, 2, 2, 5, 5, 5, 5, 9, 9, 9, 12, 13]

        self.scanned_subnets = set()
        self.scanned_ips = set()
        self.exploited_ips = set()
        self.escalated_hosts = set()
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None

    def train(self, results: Results):
        """allows an agent to learn a policy"""
        pass

    def get_action(self, observation, action_space):
        self._process_success(observation)
        """gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space"""
        session = 0

        while True:
            if observation["success"] == TrinaryEnum.TRUE:
                self.action += 1 if self.action < 14 else 0
            else:
                self.action = self.jumps[self.action]

            if self.action in self.action_history:
                action = self.action_history[self.action]

            # Discover Remote Systems
            elif self.action == 0:
                self.initial_ip = observation["User0"]["Interface"][0]["IP Address"]
                self.last_subnet = observation["User0"]["Interface"][0]["Subnet"]
                self.scanned_subnets.add(self.last_subnet)
                action = DiscoverRemoteSystems(
                    session=session, agent="Red", subnet=self.last_subnet
                )
            # Discover Network Services- new IP address found
            elif self.action == 1:
                hosts = [
                    value
                    for key, value in observation.items()
                    if key not in ["success", "Active_Sessions"]
                ]

                def get_ip(x):
                    return x["Interface"][0]["IP Address"]

                interfaces = [get_ip(x) for x in hosts if get_ip(x) != self.initial_ip]
                self.last_ip_address = random.choice(interfaces)
                # self.last_ip_address = interfaces[1]
                self.scanned_ips.add(self.last_ip_address)
                action = DiscoverNetworkServices(
                    session=session, agent="Red", ip_address=self.last_ip_address
                )

            # Exploit User1
            elif self.action == 2:
                self.exploited_ips.add(self.last_ip_address)
                action = ExploitRemoteService(
                    session=session, agent="Red", ip_address=self.last_ip_address
                )

            # Privilege escalation on User Host
            elif self.action == 3:
                hostname = [
                    value
                    for key, value in observation.items()
                    if key not in ["success", "Active_Sessions"]
                    and "System info" in value
                ][0]["System info"]["Hostname"]
                self.escalated_hosts.add(hostname)
                action = PrivilegeEscalate(
                    agent="Red", hostname=hostname, session=session
                )

            # Discover Network Services- new IP address found
            elif self.action == 4:
                self.enterprise_host = [x for x in observation if "Enterprise" in x]
                if len(self.enterprise_host) > 0:
                    self.enterprise_host = self.enterprise_host[0]
                    self.last_ip_address = observation[self.enterprise_host][
                        "Interface"
                    ][0]["IP Address"]
                    self.scanned_ips.add(self.last_ip_address)
                    action = DiscoverNetworkServices(
                        session=session, agent="Red", ip_address=self.last_ip_address
                    )
                else:
                    self.action = 3
                    action = self.action_history[self.action]
            # Exploit- Enterprise Host
            elif self.action == 5:
                self.target_ip_address = [
                    value
                    for key, value in observation.items()
                    if key not in ["success", "Active_Sessions"]
                ][0]["Interface"][0]["IP Address"]
                self.exploited_ips.add(self.target_ip_address)
                action = ExploitRemoteService(
                    session=session, agent="Red", ip_address=self.target_ip_address
                )

            # Privilege escalation on Enterprise Host
            elif self.action == 6:
                hostname = [
                    value
                    for key, value in observation.items()
                    if key not in ["success", "Active_Sessions"]
                    and "System info" in value
                ][0]["System info"]["Hostname"]
                self.escalated_hosts.add(hostname)
                action = PrivilegeEscalate(
                    agent="Red", hostname=hostname, session=session
                )

            # Scanning the new subnet found.
            elif self.action == 7:
                self.last_subnet = observation[self.enterprise_host]["Interface"][0][
                    "Subnet"
                ]
                self.scanned_subnets.add(self.last_subnet)
                action = DiscoverRemoteSystems(
                    subnet=self.last_subnet, agent="Red", session=session
                )

            # Discover Network Services- Enterprise2
            elif self.action == 8:
                self.target_ip_address = [
                    value
                    for key, value in observation.items()
                    if key not in ["success", "Active_Sessions"]
                ][2]["Interface"][0]["IP Address"]
                self.scanned_ips.add(self.target_ip_address)
                action = DiscoverNetworkServices(
                    session=session, agent="Red", ip_address=self.target_ip_address
                )

            # Exploit- Enterprise2
            elif self.action == 9:
                self.target_ip_address = [
                    value
                    for key, value in observation.items()
                    if key not in ["success", "Active_Sessions"]
                ][0]["Interface"][0]["IP Address"]
                self.exploited_ips.add(self.target_ip_address)
                action = ExploitRemoteService(
                    session=session, agent="Red", ip_address=self.target_ip_address
                )

            # Privilege escalation on Enterprise2
            elif self.action == 10:
                hostname = [
                    value
                    for key, value in observation.items()
                    if key not in ["success", "Active_Sessions"]
                    and "System info" in value
                ][0]["System info"]["Hostname"]
                self.escalated_hosts.add(hostname)
                action = PrivilegeEscalate(
                    agent="Red", hostname=hostname, session=session
                )

            # Discover Network Services- Op_Server0
            elif self.action == 11:
                self.scanned_ips.add(
                    observation["Op_Server0"]["Interface"][0]["IP Address"]
                )
                action = DiscoverNetworkServices(
                    session=session,
                    agent="Red",
                    ip_address=observation["Op_Server0"]["Interface"][0]["IP Address"],
                )

            # Exploit- Op_Server0
            elif self.action == 12:
                info = [
                    value
                    for key, value in observation.items()
                    if key not in ["success", "Active_Sessions"]
                ]
                if len(info) > 0:
                    self.exploited_ips.add(info[0]["Interface"][0]["IP Address"])
                    action = ExploitRemoteService(
                        agent="Red",
                        session=session,
                        ip_address=info[0]["Interface"][0]["IP Address"],
                    )
                else:
                    self.action = 0
                    continue
            # Privilege escalation on Op_Server0
            elif self.action == 13:
                self.escalated_hosts.add("Op_Server0")
                action = PrivilegeEscalate(
                    agent="Red", hostname="Op_Server0", session=session
                )
            # Impact on Op_server0
            elif self.action == 14:
                action = Impact(agent="Red", session=session, hostname="Op_Server0")

            if self.action not in self.action_history:
                self.action_history[self.action] = action
            return action

    def _process_success(self, observation):
        if self.last_ip is not None:
            if observation["success"] == TrinaryEnum.TRUE:
                self.host_ip_map[
                    [
                        value["System info"]["Hostname"]
                        for key, value in observation.items()
                        if key != "success"
                        and "System info" in value
                        and "Hostname" in value["System info"]
                    ][0]
                ] = self.last_ip
            else:
                self._process_failed_ip()
            self.last_ip = None
        if self.last_host is not None:
            if observation["success"] == TrinaryEnum.FALSE:
                if self.last_host in self.escalated_hosts:
                    self.escalated_hosts.remove(self.last_host)
                if (
                    self.last_host in self.host_ip_map
                    and self.host_ip_map[self.last_host] in self.exploited_ips
                ):
                    self.exploited_ips.remove(self.host_ip_map[self.last_host])
            self.last_host = None

    def _process_failed_ip(self):
        self.exploited_ips.remove(self.last_ip)

        def hosts_of_type(y):
            return [x for x in self.escalated_hosts if y in x]

        if len(hosts_of_type("Op")) > 0:
            for host in hosts_of_type("Op"):
                self.escalated_hosts.remove(host)
                ip = self.host_ip_map[host]
                self.exploited_ips.remove(ip)
        elif len(hosts_of_type("Ent")) > 0:
            for host in hosts_of_type("Ent"):
                self.escalated_hosts.remove(host)
                ip = self.host_ip_map[host]
                self.exploited_ips.remove(ip)

    def end_episode(self):
        self.action = 0
        self.target_ip_address = None
        self.last_subnet = None
        self.last_ip_address = None
        self.action_history = {}

        self.scanned_subnets = set()
        self.scanned_ips = set()
        self.exploited_ips = set()
        self.escalated_hosts = set()
        self.host_ip_map = {}
        self.last_host = None
        self.last_ip = None

    def set_initial_values(self, action_space, observation):
        pass
