from argparse import ArgumentParser
from statistics import mean, stdev
from typing import Union
import random
import numpy as np

from joblib import Parallel, delayed
import torch

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent

from CardiffUni.Agents.MainAgent import MainAgent

from models.ppo import GraphPPOAgent
from models.inductive_ppo import load_inductive_ppo
from models.red_inductive_ppo import load_inductive_ppo as load_red_inductive_ppo
from wrapper.red_blue_wrapper import RedBlueWrapper

from adapters.BlueGraphAdapter import BlueGraphAdapter
from CybORG.Shared.Actions import Restore
from adapters.BlueChallengeAdapter import BlueChallengeAdapter
from adapters.RedGraphAdapter import RedGraphAdapter
from rewards.cc2_subtask_functions import (
    subtask_1_complete,
    subtask_1_port_connected,
    subtask_2_complete,
    subtask_2_port_connected,
    flood_check,
    impact_complete
)
from rewards.bt import BehaviourTreeConfig


MAX_EPS = 1000
SAVE_ACTION_DISTRO = False
SHUFFLE_NIDS = False
NODE_ACTION_SPACE = 11
N_JOBS = 30


@torch.no_grad()
def generate_episode(
    agent,
    red_agent: Union[GraphPPOAgent, type[BaseAgent]],
    num_nodes,
    add_flood_actions=False,
    add_stealth_action=False,
    cc2_reward=False,
    num_steps=100,
    exclude_bt_phase=False,
    use_cardiff_adapter=False,
    seed=0,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cyborg = CybORG(path, "parallel_sim")
    if not isinstance(red_agent, GraphPPOAgent):
        red_agent_object = red_agent()
        add_flood_actions = False
        add_stealth_action = False
        cc2_reward = True
    bt_config = None
    flood_k = None
    flood_node_ids = None
    if not cc2_reward:
        if add_flood_actions:
            if "cage2" in path or "extra_ent_op" in path:
                flood_k = 3
            else:
                flood_k = 6
            flood_node_ids = range(num_nodes)
        bt_config = BehaviourTreeConfig(
            subtask_names=["To_Enterprise", "To_Op_Server"],
            subtask_complete_functions=[subtask_1_complete, subtask_2_complete],
            subtask_port_functions=[subtask_1_port_connected, subtask_2_port_connected],
            subtask_flood_function=flood_check,
            mission_complete_function=impact_complete,
            use_memory=False,
            flood_k=flood_k,
            flood_node_ids=flood_node_ids,
        )

    if not use_cardiff_adapter:
        blue_adapter = BlueGraphAdapter(
            "Blue", cyborg, sample_result=cyborg.reset_parallel(["Blue"])["Blue"]
        )
    else:
        blue_adapter = BlueChallengeAdapter(
            "Blue", cyborg, sample_result=cyborg.reset_parallel(["Blue"])["Blue"]
        )
    red_adapter = RedGraphAdapter(
        cyborg,
        max_nodes=num_nodes,
        bt_config=bt_config,
        reward_mode="bt_as_reward" if not cc2_reward else "environment",
        add_flood_actions=add_flood_actions,
        add_stealth_action=add_stealth_action,
        use_bt_phase=not exclude_bt_phase,
    )
    wrapped_cyborg = RedBlueWrapper(cyborg, red_adapter, blue_adapter)

    r = []
    a = []
    s = 0
    restore_actions = 0

    combined_obs, combined_info = wrapped_cyborg.reset()
    # start_time = time.time()
    for j in range(num_steps):
        blue_action = agent.get_action(combined_obs["Blue"])
        if isinstance(red_agent, GraphPPOAgent):
            red_action = red_agent.get_action(combined_obs["Red"])
        else:
            red_action = red_agent_object.get_action(
                combined_info["Red"]["observation"],
                combined_info["Red"]["action_space"],
            )
        next_combined_obs, combined_rew, combined_done, combined_truncated, combined_info = (
            wrapped_cyborg.step(red_action_int=red_action, blue_action_int=blue_action)
        )
        
        if combined_info["Red"]["success"]:
            s += 1
        if isinstance(combined_info["Blue"]["cyborg_action"], Restore):
            restore_actions += 1
        r.append(combined_rew["Blue"])
        a.append(
            (str(cyborg.get_last_action("Blue")), str(cyborg.get_last_action("Red")))
        )
        
        combined_obs = next_combined_obs
    # print(f"Episode finished in {time.time() - start_time:.2f} seconds")
    agent.end_episode()
    return sum(r), s, restore_actions


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("-s", "--scenario", default="scenarios/cage2.yaml")
    ap.add_argument("--blue_checkpoint", default=None, type=str)
    ap.add_argument("--red_checkpoint", default=None, type=str)
    ap.add_argument("--globalnode", action="store_true")
    ap.add_argument("--cardiff", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--num_nodes", default=13, type=int)
    ap.add_argument("--add_flood_actions", action="store_true")
    ap.add_argument("--add_stealth_action", action="store_true")
    ap.add_argument("--cc2_reward", action="store_true")
    ap.add_argument("--exclude_bt_phase", action="store_true")
    ap.add_argument("--fname", default="results", type=str)
    args = ap.parse_args()
    
    if args.blue_checkpoint is not None:
        agent = load_inductive_ppo(
            args.blue_checkpoint,
            globalnode=True,
            num_nodes=args.num_nodes,
        )
        agent.set_deterministic(True)
    # KEEP
    elif args.globalnode:
        agent = load_inductive_ppo(
            "model_weights/inductive_global-attn.pt",
            globalnode=True,
            num_nodes=args.num_nodes,
        ) 
        agent.set_deterministic(True)
    # Cardiff
    elif args.cardiff:
        agent = MainAgent()
    else:
        raise ValueError("Unknown or unspecified model type")

    path = args.scenario  # 'scenarios/wolk_et_al/scenario3.yaml'

    if args.red_checkpoint is not None:
        r_agents = [
            load_red_inductive_ppo(
                args.red_checkpoint, globalnode=True, num_nodes=args.num_nodes
            )
        ]
    else:
        r_agents = [B_lineAgent, RedMeanderAgent]

    # print(generate_episode(agent, 
    #                         r_agents[0],
    #                         add_flood_actions=args.add_flood_actions, 
    #                         add_stealth_action=args.add_stealth_action, 
    #                         num_nodes=args.num_nodes,
    #                         num_steps=100,
    #                         use_cardiff_adapter=args.cardiff,)
    # )
    
    import os
    # Create results/red directory if it doesn't exist
    os.makedirs("results/red", exist_ok=True)

    all_r = []
    impact_r = []
    for num_steps in [100]:
        for red_agent in r_agents:
            res = Parallel(n_jobs=N_JOBS, prefer="processes")(
                delayed(generate_episode)(
                    agent=agent,
                    red_agent=red_agent,
                    num_nodes=args.num_nodes,
                    add_flood_actions=args.add_flood_actions,
                    add_stealth_action=args.add_stealth_action,
                    cc2_reward=args.cc2_reward,
                    exclude_bt_phase=args.exclude_bt_phase,
                    num_steps=num_steps,
                    use_cardiff_adapter=args.cardiff,
                    seed=i,
                )
                for i in range(MAX_EPS)
            )
            total_reward, success_count, restore_actions = zip(*res)

            r_agent_name = (
                red_agent.__name__
                if not isinstance(red_agent, GraphPPOAgent)
                else red_agent.__class__.__name__
            )

            print(
                f"Average reward for red agent {r_agent_name} and steps {num_steps} is: {mean(total_reward)} with a standard deviation of {stdev(total_reward)}, {mean(success_count)}, {stdev(success_count)}, {max(success_count)}"
            )
            # print(
            #     f"Average restore actions for red agent {r_agent_name} and steps {num_steps} is: {mean(restore_actions)} with a standard deviation of {stdev(restore_actions)}, {max(restore_actions)}"
            # )

            all_r.extend(total_reward)
            impact_r.extend(success_count)

    with open(f"results/red/{args.fname}.csv", "w") as f:
        # Write blue rewards and red impact to csv
        f.write("blue_reward,red_impact\n")
        for r, impact in zip(all_r, impact_r):
            f.write(f"{r},{impact}\n")
    print()
