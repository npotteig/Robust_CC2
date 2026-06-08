## The following code contains work of the United States Government and is not subject to domestic copyright protection under 17 USC § 105.
## Additionally, we waive copyright and related rights in the utilized code worldwide through the CC0 1.0 Universal public domain dedication.

import sys
import yaml
from typing import Dict, List

from CybORG.Shared import Scenario
from CybORG.Shared.Actions.Action import InvalidAction
from CybORG.Shared.Results import Results
from CybORG.Shared.Observation import Observation
from CybORG.Shared.Actions import Action, Monitor, AddHost
from CybORG.Shared.AgentInterface import AgentInterface


class ParallelEnvironmentController:
    """The abstract base controller for all CybORG environment controllers.

    Provides the abstract methods which all CybORG controllers must implement. This includes setup and teardown,
    modifying the state, and pulling out data from the environment.
    When both Simulation and Emulation share common functionality, it is implemented here.


    Attributes
    ----------
    scenario_dict : dict
        the scenario data
    agent_interfaces : dict[str: AgentInterface]
        agent interface object for agents in scenario
    """

    def __init__(
        self, scenario_path: str, scenario_mod: dict = None, agents: dict = None
    ):
        """Instantiates the Environment Controller.
        Parameters
        ----------
        scenario_path : str
            path to scenario YAML file
        agents : dict, optional
            map from agent name to agent interface of agents to be used in
            environment. If None agents will be loaded from description in
            scenario file (default=None)
        """
        self.hostname_ip_map = None
        self.subnet_cidr_map = None
        # self.scenario_dict = self._parse_scenario(scenario_path, scenario_mod=scenario_mod)
        scenario_dict = self._parse_scenario(scenario_path)
        self.scenario = Scenario(scenario_dict)
        self._create_environment()
        self.agent_interfaces = self._create_agents(agents)
        self.reward = {}
        self.INIT_INFO_DICT = {}
        self.action = {}
        self.done = False
        self.observation = {}
        self.INIT_INFO_DICT["True"] = {}
        for host in self.scenario.hosts:
            self.INIT_INFO_DICT["True"][host] = {
                "System info": "All",
                "Sessions": "All",
                "Interfaces": "All",
                "User info": "All",
                "Processes": ["All"],
            }
        self.init_state = self._filter_obs(
            self.get_true_state(self.INIT_INFO_DICT["True"])
        ).data
        for agent in self.scenario.agents:
            self.INIT_INFO_DICT[agent] = self.scenario.get_agent_info(agent).osint.get(
                "Hosts", {}
            )
            for host in self.INIT_INFO_DICT[agent].keys():
                self.INIT_INFO_DICT[agent][host]["Sessions"] = agent
        # populate initial observations with OSINT
        for agent_name, agent in self.agent_interfaces.items():
            self.observation[agent_name] = self._filter_obs(
                self.get_true_state(self.INIT_INFO_DICT[agent_name]), agent_name
            )
            agent.set_init_obs(self.observation[agent_name].data, self.init_state)

        self.INFO_DICT = self.INIT_INFO_DICT.copy()

    def reset(self, agents: List[str] = None) -> Dict[str, Results]:
        """Resets the environment and get initial agent observation and actions.

        Parameters
        ----------
        agents : List[str], optional
            the agents to get initial observation for, if None will return
            initial white state (default=None)

        Returns
        -------
        Dict[str, Results]
            The initial observation and actions of agents or white team
        """
        self.reward = {}
        self.steps = 0
        self.done = False
        self.INFO_DICT = self.INIT_INFO_DICT.copy()
        self.init_state = self._filter_obs(
            self.get_true_state(self.INFO_DICT["True"])
        ).data
        for agent_name, agent_object in self.agent_interfaces.items():
            agent_object.reset()
            self.observation[agent_name] = self._filter_obs(
                self.get_true_state(self.INFO_DICT[agent_name]), agent_name
            )
            agent_object.set_init_obs(
                self.observation[agent_name].data, self.init_state
            )

        # Update Red's observation with active red sessions
        self.observation["Red"].data["Active_Sessions"] = {}
        for state_key, state_object in self.init_state.items():
            if state_key != "success" and "Sessions" in state_object:
                for session in state_object["Sessions"]:
                    if session["Agent"] == "Red":
                        self.observation["Red"].data["Active_Sessions"][state_key] = (
                            session
                        )

        if agents is None:
            return {"all": Results(observation=self.init_state)}
        else:
            return {
                agent: Results(
                    observation=self.observation[agent].data,
                    action_space=self.agent_interfaces[
                        agent
                    ].action_space.get_action_space(),
                )
                for agent in agents
            }

    def step(
        self, actions: Dict[str, Action] = None, skip_valid_action_check: bool = False
    ) -> Dict[str, Results]:
        """Perform a step in the environment for given agent.

        Parameters
        ----------
        actions : Dict[str, Action], optional
            a dictionary mapping agent names to their actions (default=None)
        skip_valid_action_check : bool, optional
            whether to skip the validation of actions (default=False)

        Returns
        -------
        Dict[str, Results]
            the result of agents performing the action
        """

        # for each agent:
        next_observation = {}
        # all agents act on the state
        for agent_name, agent_object in self.agent_interfaces.items():
            # pass observation to agent to get action
            if (
                actions is None
                or agent_name not in actions
                or actions[agent_name] is None
            ):
                if agent_name == "Red":
                    agent_action = [
                        agent_object.get_action(self.observation[agent_name])
                    ]
                else:
                    agent_action = agent_object.get_action(self.observation[agent_name])
            else:
                agent_action = actions[agent_name]
            if agent_name == "Red":
                for act in agent_action:
                    if (
                        not self.test_valid_action(act, agent_object)
                        and not skip_valid_action_check
                    ):
                        agent_action = InvalidAction(act)
                    self.action[agent_name] = act

                    executed_action_obs = self.execute_action(self.action[agent_name])
            else:
                if (
                    not self.test_valid_action(agent_action, agent_object)
                    and not skip_valid_action_check
                ):
                    agent_action = InvalidAction(agent_action)
                self.action[agent_name] = agent_action
                executed_action_obs = self.execute_action(self.action[agent_name])
            if isinstance(agent_action, AddHost) and executed_action_obs.success:
                self.INFO_DICT["True"][agent_action.hostname] = agent_action.INFO_DICT[
                    agent_action.hostname
                ]
                self.INFO_DICT["Blue"][agent_action.hostname] = (
                    agent_action.sim_mod.scenario.get_agent_info("Blue").osint["Hosts"][
                        agent_action.hostname
                    ]
                )
                self.INFO_DICT["Green"][agent_action.hostname] = (
                    agent_action.sim_mod.scenario.get_agent_info("Green").osint[
                        "Hosts"
                    ][agent_action.hostname]
                )
                self.INFO_DICT["Blue"][agent_action.hostname]["Sessions"] = "Blue"
                self.INFO_DICT["Green"][agent_action.hostname]["Sessions"] = "Green"
                self.scenario._scenario["Hosts"][agent_action.hostname] = (
                    agent_action.sim_mod.scenario._scenario["Hosts"][
                        agent_action.hostname
                    ]
                )
                self.hostname_ip_map = {
                    h: ip for ip, h in self.state.ip_addresses.items()
                }
            # perform action on state
            next_observation[agent_name] = self._filter_obs(
                executed_action_obs, agent_name
            )

        # get true observation
        true_observation = self._filter_obs(
            self.get_true_state(self.INFO_DICT["True"])
        ).data

        # Blue update step.
        # New idea: run the MONITOR action for the Blue agent, and update the observation.

        # pass training information to agents
        for agent_name, agent_object in self.agent_interfaces.items():
            # determine done signal for agent
            done = self.determine_done(
                next_observation, true_observation, self.action[agent_name]
            )
            self.done = done or self.done
            # determine reward for agent
            reward = agent_object.determine_reward(
                next_observation, true_observation, self.action, self.done
            )
            self.reward[agent_name] = reward + self.action[agent_name].cost
            if agent_name not in actions:
                # train agent using obs, reward, previous observation, and done
                agent_object.train(
                    Results(
                        observation=self.observation[agent_name].data,
                        reward=reward,
                        next_observation=next_observation[agent_name].data,
                        done=self.done,
                    )
                )
            self.observation[agent_name] = next_observation[agent_name]
            agent_object.update(self.observation[agent_name])

            # if self.verbose and type(self.action[agent_name]) != Sleep and self.observation[agent_name].dict['success'] == True:
            #    print(f"Step: {self.steps}, {agent_name}'s Action Choice: {type(self.action[agent_name]).__name__}, "
            #          f"Reward: {reward}")
        # Update Red's observation with active red sessions
        self.observation["Red"].data["Active_Sessions"] = {}
        for state_key, state_object in true_observation.items():
            if state_key != "success" and "Sessions" in state_object:
                for session in state_object["Sessions"]:
                    if session["Agent"] == "Red":
                        self.observation["Red"].data["Active_Sessions"][state_key] = (
                            session
                        )

        if (
            "Green" in actions
            and isinstance(actions["Green"], AddHost)
            and self.observation["Green"].success
        ):
            blue_obs = {
                actions["Green"].hostname: self._filter_obs(
                    self.get_true_state(self.INFO_DICT["Blue"]), "Blue"
                ).data[actions["Green"].hostname]
            }
            self.agent_interfaces["Blue"].update(blue_obs, known=True)
            self.observation["Blue"].data["NEW_HOST"] = blue_obs
            self.agent_interfaces["Red"].update(
                {
                    actions["Green"].hostname: true_observation[
                        actions["Green"].hostname
                    ]
                },
                known=False,
                new_keys_only=True,
            )

        # Update Blue's observation with the latest information before returning.
        for agent_name, agent_object in self.agent_interfaces.items():
            if agent_name == "Blue":
                agent_session = list(
                    self.get_action_space(agent_name)["session"].keys()
                )[0]
                agent_observation = self._filter_obs(
                    self.execute_action(Monitor(session=agent_session, agent="Blue")),
                    agent_name,
                )
                first_action_success = self.observation[agent_name].success
                self.observation[agent_name].combine_obs(agent_observation)
                self.observation[agent_name].set_success(first_action_success)
                agent_object.update(self.observation[agent_name])
        # if done then complete other agent's turn

        if actions is None:
            result = {"all": Results(observation=true_observation, done=self.done)}
        else:
            result = {
                agent: Results(
                    observation=self.observation[agent].data,
                    done=self.done,
                    reward=round(self.reward[agent], 1),
                    action_space=self.agent_interfaces[
                        agent
                    ].action_space.get_action_space(),
                    action=self.action[agent],
                )
                for agent in actions
            }
        return result

    def execute_action(self, action: Action) -> Observation:
        """Execute an action in the environment"""
        raise NotImplementedError

    def determine_done(self, agent_obs: dict, true_obs: dict, action: Action) -> bool:
        """Determine if environment scenario goal has been reached.

        Parameters
        ----------
        agent_obs : dict
            the agents last observation
        true_obs : dict
            the current white state
        action : Action
            the agents last action performed

        Returns
        -------
        bool
            whether goal was reached or not
        """
        return False

    def start(self, steps: int = None, log_file=None):
        """Start the environment and run for a specified number of steps.

        Parameters
        ----------
        steps : int
            the number of steps to run for
        log_file : File, optional
            a file to write results to (default=None)

        Returns
        -------
        bool
            whether goal was reached or not
        """
        done = False
        max_steps = 0
        if steps is None:
            while not done:
                max_steps += 1
                _, _, done = self.step()
            print("Red Wins!")  # Junk Test Code
        else:
            for step in range(steps):
                max_steps += 1
                results = self.step()
                done = results.done
                if step == 500:
                    print(step)  # Junk Test Code
                if done:
                    print(f"Red Wins at step {step}")  # Junk Test Code
                    break
        for agent_name, agent in self.agent_interfaces.items():
            agent.end_episode()
            # print(f"{agent_name}'s Reward: {self.reward[agent_name]}")
        if log_file is not None:
            log_file.write(
                f"{max_steps},{self.reward['Red']},{self.reward['Blue']},"
                f"{self.agent_interfaces['Red'].agent.epsilon},"
                f"{self.agent_interfaces['Red'].agent.gamma}\n"
            )
        return done

    def get_true_state(self, info: dict) -> Observation:
        """Get current True state

        Returns
        -------
        Observation
            current true state
        """
        raise NotImplementedError

    def get_agent_state(self, agent_name: str) -> Observation:
        return self.get_true_state(self.INFO_DICT[agent_name])

    def get_last_observation(self, agent: str) -> Observation:
        """Get the last observation for an agent

        Parameters
        ----------
        agent : str
            name of agent to get observation for

        Returns
        -------
        Observation
            agents last observation
        """
        return self.observation[agent]

    def get_action_space(self, agent: str) -> dict:
        """
        Gets the action space for a chosen agent
        agent: str
            agent selected
        """
        if agent in self.agent_interfaces:
            return self.agent_interfaces[agent].action_space.get_action_space()
        raise ValueError(
            f"Agent {agent} not in agent list {self.agent_interfaces.values()}"
        )

    def get_observation_space(self, agent: str) -> dict:
        """
        Gets the observation space for a chosen agent
        agent: str
            agent selected
        """
        if agent in self.agent_interfaces:
            return self.agent_interfaces[agent].get_observation_space()
        raise ValueError(
            f"Agent {agent} not in agent list {self.agent_interfaces.values()}"
        )

    def get_last_action(self, agent: str) -> Action:
        """
        Gets the observation space for a chosen agent
        agent: str
            agent selected
        """
        return self.action[agent] if agent in self.action else None

    def restore(self, filepath: str):
        """Restores the environment from file

        Parameters
        ----------
        filepath : str
            path to file to restore env from
        """
        raise NotImplementedError

    def save(self, filepath: str):
        """Saves the environment to file

        Parameters
        ----------
        filepath : str
            path to file to save env to
        """
        raise NotImplementedError

    def pause(self):
        """Pauses the environment"""
        pass

    def shutdown(self, teardown: bool = True) -> bool:
        """Shutdown environment, deleting/terminating resources
        as required

        Parameters
        ----------
        teardown : bool, optional
            if True environment resources will be terminated if applicable,
            otherwise resources will not be terminated (allowing them to be
            reused if desired) (default=True)

        Returns
        -------
        bool
            True if the environment was shutdown without issue
        """
        raise NotImplementedError

    def _parse_scenario(self, scenario_file_path: str, scenario_mod: dict = None):
        with open(scenario_file_path) as fIn:
            scenario_dict = yaml.load(fIn, Loader=yaml.FullLoader)
        return scenario_dict

    def _create_agents(self, agent_classes: dict = None) -> dict:
        agents = {}

        for agent_name in self.scenario.agents:
            agent_info = self.scenario.get_agent_info(agent_name)
            if agent_classes is not None and agent_name in agent_classes:
                agent_class = agent_classes[agent_name]
            else:
                agent_class = getattr(
                    sys.modules["CybORG.Agents"], agent_info.agent_type
                )
            agents[agent_name] = AgentInterface(
                agent_class,
                agent_name,
                agent_info.actions,
                agent_info.reward_calculator_type,
                allowed_subnets=agent_info.allowed_subnets,
                wrappers=agent_info.wrappers,
                scenario=self.scenario,
            )
        return agents

    def _create_environment(self):
        raise NotImplementedError

    def _filter_obs(self, obs: Observation, agent_name=None):
        """Filter obs to contain only hosts/subnets in scenario network"""
        if agent_name is not None:
            subnets = [
                self.subnet_cidr_map[s]
                for s in self.scenario.get_agent_info(agent_name).allowed_subnets
            ]
        else:
            subnets = list(self.subnet_cidr_map.values())

        obs.filter_addresses(
            ips=self.hostname_ip_map.values(), cidrs=subnets, include_localhost=False
        )
        return obs

    def test_valid_action(self, action: Action, agent: AgentInterface):
        # returns true if the parameters in the action are in and true in the action set else return false
        action_space = agent.action_space.get_action_space()
        # first check that the action class is allowed
        if (
            type(action) not in action_space["action"]
            or not action_space["action"][type(action)]
        ):
            return False
        # next for each parameter in the action
        for parameter_name, parameter_value in action.get_params().items():
            if parameter_name not in action_space:
                continue
            if not isinstance(action, AddHost) and (
                parameter_value not in action_space[parameter_name]
                or not action_space[parameter_name][parameter_value]
            ):
                return False
        return True

    def get_reward_breakdown(self, agent: str):
        return self.agent_interfaces[agent].reward_calculator.host_scores
