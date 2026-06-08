from __future__ import annotations
from CybORG.Shared.Actions import Action
from CybORG.Shared import Observation
from CybORG.Simulator.State import State
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from CybORG.Simulator.SimulationController import SimulationMod


class AddHost(Action):
    def __init__(self, hostname: str, sim_mod: SimulationMod):
        super().__init__()
        self.hostname = hostname
        self.sim_mod = sim_mod
        self.INFO_DICT = {}
        for host in self.sim_mod.scenario.hosts:
            self.INFO_DICT[host] = {
                "System info": "All",
                "Sessions": "All",
                "Interfaces": "All",
                "User info": "All",
                "Processes": ["All"],
            }

    def sim_execute(self, state: State):
        if self.hostname not in self.sim_mod.scenario.hosts:
            return Observation(success=False)
        if self.hostname in state.hosts:
            return Observation(success=True)

        # Add Host
        state.hosts[self.hostname] = self.sim_mod.state.hosts[self.hostname]
        # Add IP Address
        state.ip_addresses[self.sim_mod.hostname_ip_map[self.hostname]] = self.hostname
        # Update IP Address in subnet
        state.subnets[
            self.sim_mod.state.hosts[self.hostname].interfaces[1].subnet
        ].ip_addresses.append(self.sim_mod.hostname_ip_map[self.hostname])
        # Add starting session
        for agent in self.sim_mod.state.sessions:
            for session_id, session in self.sim_mod.state.sessions[agent].items():
                if session.host == self.hostname:
                    state.sessions[agent][session_id] = session
                    state.sessions_count[agent] += 1

                    # Link parent if exists
                    if session.parent is not None:
                        state.sessions[agent][session.parent].children[session_id] = (
                            session
                        )

        obs = state.get_true_state({self.hostname: self.INFO_DICT[self.hostname]})
        obs.set_success(True)
        return obs

    def __str__(self):
        return f"{self.__class__.__name__} {self.hostname}"
