from ipaddress import IPv4Address
from typing import Optional

from CybORG.Shared.Actions.ConcreteActions.ConcreteAction import ConcreteAction
from CybORG.Shared.Actions.MSFActionsFolder.MSFAction import lo
from CybORG.Shared.Enums import ProcessType
from CybORG.Shared.Observation import Observation
from CybORG.Simulator.Host import Host
from CybORG.Simulator.State import State
from CybORG.Simulator.Process import Process


class FloodPort(ConcreteAction):
    def __init__(self, session: int, agent: str, ip_address: IPv4Address):
        super().__init__(session=session, agent=agent)
        self.ip_address = ip_address
        self.session = session
        self.agent = agent
        self.length_of_wordlist = 3  # Reduced from 10 to save memory during training
        # self.seq_of_ports = [22, 445, 25, 80] # ssh, smb, smtp, apache2 webserver
        # self.seq_of_proc_types = [ProcessType.SSH, ProcessType.SMB, ProcessType.SMTP, ProcessType.WEBSERVER]
        # self.port_to_proc_type = {22: ("ssh", ProcessType.SSH), 445: ("smb", ProcessType.SMB), 25: ("smtp", ProcessType.SMTP), 80: ("webserver", ProcessType.WEBSERVER)}
        self.proc_type_to_port = {
            ProcessType.SSH: 22,
            ProcessType.SMB: 445,
            ProcessType.SMTP: 25,
            ProcessType.WEBSERVER: 80,
        }
        self.proc_type_to_name = {
            ProcessType.SSH: "ssh",
            ProcessType.SMB: "smb",
            ProcessType.SMTP: "smtp",
            ProcessType.WEBSERVER: "webserver",
        }

    def sim_execute(self, state: State):
        self.state = state
        obs = Observation()
        if self.session not in state.sessions[self.agent]:
            obs.set_success(False)
            return obs
        from_host: Host = state.hosts[state.sessions[self.agent][self.session].host]
        session = state.sessions[self.agent][self.session]

        if not session.active:
            obs.set_success(False)
            return obs

        # determine which ports can communicate between which subnets
        originating_ip_address = None
        if self.ip_address == lo:
            target_host: Host = from_host
            originating_ip_address = self.ip_address
        else:
            target_host: Host = state.hosts[state.ip_addresses[self.ip_address]]
            ports = self.check_routable(
                [
                    state.subnets[i.subnet]
                    for i in from_host.interfaces
                    if i.ip_address != lo
                ],
                [s for s in state.subnets.values() if self.ip_address in s.cidr],
            )

        valid_port = next(
            (p for p in self.proc_type_to_port.values() if p in ports), None
        )
        if valid_port is None:
            valid_port = "all" if ports and "all" in ports else None
        if valid_port is None:
            obs.set_success(False)
            return obs
        from_subnet, to_subnet = ports[valid_port]
        # calculate the originating ip address
        for i in from_host.interfaces:
            if i.ip_address != lo:
                if i.subnet == from_subnet:
                    originating_ip_address = i.ip_address

        vuln_proc: Optional[Process] = None
        for proc in target_host.processes:
            selected_process_type = next(
                (pt for pt in self.proc_type_to_port if pt == proc.process_type), None
            )
            if selected_process_type is not None:
                for conn in proc.connections:
                    if (
                        "local_port" in conn
                        and conn["local_port"]
                        == self.proc_type_to_port[selected_process_type]
                    ):
                        vuln_proc = proc
                        break
                if vuln_proc is not None:
                    break

        if vuln_proc is None:
            obs.set_success(False)
            return obs

        obs.add_process(
            hostid=str(self.ip_address),
            local_address=self.ip_address,
            local_port=self.proc_type_to_port[vuln_proc.process_type],
            status="open",
            process_type=self.proc_type_to_name.get(vuln_proc.process_type),
        )

        for i in range(self.length_of_wordlist):
            target_host.events["NetworkConnections"].append(
                {
                    "remote_address": originating_ip_address,
                    "remote_port": from_host.get_ephemeral_port(),
                    "local_address": self.ip_address,
                    "local_port": self.proc_type_to_port[vuln_proc.process_type],
                }
            )
        obs.set_success(True)
        return obs
