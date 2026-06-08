from argparse import ArgumentParser
import inspect
import os
import time
import random
from types import SimpleNamespace
import copy

from joblib import Parallel, delayed
import torch

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent

from models.ppo import GraphPPOAgent, PPOMemory
from models.gnn_type_ablation import InductiveGraphPPOAgent
from models.inductive_ppo import load_inductive_ppo
from models.red_inductive_ppo import load_inductive_ppo as load_red_inductive_ppo

from wrapper.graph_wrapper import GraphWrapper
from wrapper.observation_graph import ObservationGraph
from wrapper.red_blue_wrapper import RedBlueWrapper

from adapters.BlueGraphAdapter import BlueGraphAdapter
from adapters.RedGraphAdapter import RedGraphAdapter
from rewards.cc2_subtask_functions import (
    subtask_1_complete,
    subtask_1_port_connected,
    subtask_2_complete,
    subtask_2_port_connected,
    flood_check,
    impact_complete
)
from rewards.bt import BehaviourTreeConfig, BehaviourTreeReward

DEFAULT_DIM = ObservationGraph.DIM + 4

N_JOBS = 25
N_THREADS = 32

torch.set_num_threads(N_THREADS)
HYPER_PARAMS = SimpleNamespace(
    N=25,  # How many episodes before training
    bs=625,  # How many steps to learn from at a time
    episode_len=100,
    training_episodes=10_001,
    epochs=4,
    eval_every=10,
    eval_episodes=25,
    scenario="scenarios/cage2.yaml",
    new_scenario=None,
    new_red_checkpoint=None,
    new_num_nodes=13,
    switch_condition_extended=lambda x: x <= 1.3,
    switch_condition_hybrid=lambda x: x >= 1.3,
    num_nodes=13,
    red_add_flood_actions=False,
    red_add_stealth_action=False,
    red_sparse_reward=False,
)
GENERATE_VIZ_DATA = False


@torch.no_grad()
def generate_episode(blue_agent, 
                     rl_red_agent,
                     scenario, 
                     num_nodes,
                     hp,
                     eval=False):
    torch.set_num_threads(1)

    cyborg = CybORG(scenario, "parallel_sim")
    red_agent = random.choice([RedMeanderAgent, B_lineAgent, rl_red_agent]) if rl_red_agent else random.choice([RedMeanderAgent, B_lineAgent])
    mem = PPOMemory(0)
    
    if not isinstance(red_agent, GraphPPOAgent):
        red_agent = red_agent()
        add_flood_actions = False
        add_stealth_action = False
    else:
        add_flood_actions = hp.red_add_flood_actions
        add_stealth_action = hp.red_add_stealth_action
    bt_config = None
    flood_k = None
    flood_node_ids = None
    if not hp.red_sparse_reward:
        if add_flood_actions:
            if "max_users" in scenario:
                flood_k = 6
            else:
                flood_k = 3
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
    blue_graph_adapter = BlueGraphAdapter(
        "Blue", cyborg, sample_result=cyborg.reset_parallel(["Blue"])["Blue"]
    )
    red_graph_adapter = RedGraphAdapter(
        cyborg,
        max_nodes=num_nodes,
        bt_config=bt_config,
        reward_mode="bt_as_reward" if not hp.red_sparse_reward else "environment",
        add_flood_actions=add_flood_actions,
        add_stealth_action=add_stealth_action,
    )
    wrapped_cyborg = RedBlueWrapper(cyborg, red_graph_adapter, blue_graph_adapter)
    impact_counts = 0
    flooded_hosts_lengths = []
    state, info = wrapped_cyborg.reset()
    red_progress_rewards = 0
    for step_cnt in range(hp.episode_len):
        if not eval:
            blue_action, value, prob = blue_agent.get_action(state["Blue"])
        else:
            with torch.no_grad():
                blue_action = blue_agent.get_action(state["Blue"])
            value = 0.0
            prob = 1.0
        if isinstance(red_agent, GraphPPOAgent):
            red_action = red_agent.get_action(state["Red"])
        else:
            red_action = red_agent.get_action(
                info["Red"]["observation"],
                info["Red"]["action_space"],
            )
        next_state, reward, _, _, info = wrapped_cyborg.step(red_action_int=red_action, blue_action_int=blue_action)
        red_progress_rewards += reward["Red"]
        mem.remember(
            state["Blue"],
            blue_action,
            value,
            prob,
            reward["Blue"],
            0 if step_cnt < hp.episode_len - 1 else 1,
        )
        flooded_hosts_lengths.append(len(wrapped_cyborg.red_adapter.graph.flooded_nodes))
        impact_counts += int(info["Red"]["success"])
        state = next_state

    return (
        mem,
        impact_counts,
        max(flooded_hosts_lengths) if flooded_hosts_lengths else 0,
        red_progress_rewards if isinstance(red_agent, GraphPPOAgent) else None,
    )


def train(blue_agent, init_red_agent, hp):
    blue_agent.train()

    with open(f"logs/blue/{hp.fnames}.txt", "w+") as f:
        f.write("epoch,avg_r,loss,avg_impact_actions,max_flooded,avg_red_progress_reward,iter_time\n")
    current_scenario = hp.scenario
    current_num_nodes = hp.num_nodes
    
    red_agent = init_red_agent
    
    for e in range(1, hp.training_episodes):
            
        start_time = time.time()
        ret = Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(generate_episode)(blue_agent, red_agent, current_scenario, current_num_nodes, hp) for _ in range(hp.N)
        )
        mems, impact_counts, flooded_host_lengths, red_progress_rewards = zip(*ret)
        mem = PPOMemory(hp.bs).load(mems)
        blue_agent.memory = mem
        avg_r = sum(mem.r) / hp.N
        avg_max_flooded = sum(flooded_host_lengths) / hp.N
        avg_impact_actions = sum(impact_counts) / hp.N
        rew_from_rl_red = [r for r in red_progress_rewards if r is not None]
        avg_red_progress_reward = sum(rew_from_rl_red) / len(rew_from_rl_red) if len(rew_from_rl_red) > 0 else 0

        torch.set_num_threads(N_THREADS)
        loss = blue_agent.learn(verbose=False)
        iter_time = time.time() - start_time
        print(
            f"[{e}] Average reward: {avg_r}, Average impact actions: {avg_impact_actions}, Average flooded hosts: {avg_max_flooded}, Average red progress reward: {avg_red_progress_reward}"
        )

        with open(f"logs/blue/{hp.fnames}.txt", "a") as f:
            f.write(f"{e},{avg_r},{loss},{avg_impact_actions},{avg_max_flooded},{avg_red_progress_reward},{iter_time}\n")

        # if hp.new_scenario and e % hp.eval_every == 0:
        #     blue_agent.eval()
        #     blue_agent.actor.num_nodes = EXTENDED_NUM_NODES
        #     blue_agent.critic.num_nodes = EXTENDED_NUM_NODES
        #     blue_agent.set_deterministic(True)
            
        #     ret = Parallel(n_jobs=hp.eval_episodes, prefer="processes")(
        #         delayed(generate_episode)(blue_agent, extended_red_agent, EXTENDED_MODEL, EXTENDED_NUM_NODES, hp, eval=True) for _ in range(hp.eval_episodes)
        #     )
            
        #     blue_agent.set_deterministic(False)
        #     blue_agent.actor.num_nodes = current_num_nodes
        #     blue_agent.critic.num_nodes = current_num_nodes
        #     blue_agent.train()
            
        #     mems, impact_counts, flooded_host_lengths, red_progress_rewards = zip(*ret)
        #     mem = PPOMemory(0).load(mems)
        #     avg_r = sum(mem.r) / hp.eval_episodes
        #     avg_max_flooded = sum(flooded_host_lengths) / hp.eval_episodes
        #     avg_impact_actions = sum(impact_counts) / hp.eval_episodes
        #     rew_from_rl_red = [r for r in red_progress_rewards if r is not None]
        #     eval_avg_red_progress_reward = sum(rew_from_rl_red) / len(rew_from_rl_red) if len(rew_from_rl_red) > 0 else 0
        #     # if len(red_progress_list) >= 4:
        #     #     red_progress_list.pop(0)
        #     # red_progress_list.append(avg_red_progress_reward)
            
        #     with open(f"logs/blue/{hp.fnames}.txt", "a") as f:
        #         f.write(f"{e}_eval,{avg_r},{loss},{avg_impact_actions},{avg_max_flooded},{eval_avg_red_progress_reward},{iter_time}\n")
        
        # if e % 5 == 0:
        #     blue_agent.save(outf=f"checkpoints/blue/{hp.fnames}/{e}.pt")
    blue_agent.save(outf=f"checkpoints/blue/{hp.fnames}.pt")


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "name",
        metavar="Name",
        action="store",
        type=str,
        nargs=1,
        help="Filename to use for output logs/weights",
    )
    ap.add_argument("-s", "--scenario", default="scenarios/cage2.yaml")
    ap.add_argument(
        "--inductive", action="store_true", help="Use naive inductive model"
    )
    ap.add_argument(
        "--attn", action="store_true", help="Use self-attention-based inductive model"
    )
    ap.add_argument(
        "--globalnode",
        action="store_true",
        help="Use global node inductive aggregation",
    )
    ap.add_argument(
        "--hidden",
        action="store",
        type=int,
        default=256,
        help="Dimension of middle layer for actor/critic",
    )
    ap.add_argument(
        "--embedding",
        action="store",
        type=int,
        default=64,
        help="Dimension of node representation for actor/critic",
    )
    ap.add_argument(
        "--layers",
        action="store",
        type=int,
        default=2,
        help="Number of layers for GCNs",
    )
    ap.add_argument("--gnn", default="gcn")
    ap.add_argument("--epochs", default=10_000, type=int)
    ap.add_argument("--use_blue_checkpoint", action="store_true")
    ap.add_argument("--blue_checkpoint", default="model_weights/inductive_global-attn.pt")
    ap.add_argument("--red_checkpoint", default=None)
    ap.add_argument("--num_nodes", default=13, type=int)
    ap.add_argument(
        "--red_add_flood_actions",
        action="store_true",
        help="Whether to add flood actions to the red action space",
    )
    ap.add_argument(
        "--red_add_stealth_action",
        action="store_true",
        help="Whether to add the stealth action to the red action space",
    )
    ap.add_argument(
        "--red_sparse_reward",
        action="store_true",
        help="Whether red should use a sparse reward (i.e. environment reward) instead of the BT reward",
    )
    
    ap.add_argument(
        "--new_scenario",
        default=None,
        help="Scenario to switch to during training",
    )
    ap.add_argument(
        "--new_red_checkpoint",
        default=None,
        help="Red agent checkpoint to switch to during training (only used if --new_scenario is set)",
    )
    ap.add_argument(
        "--new_num_nodes",
        default=13,
        type=int,
        help="Number of nodes in the new scenario (only used if --new_scenario is set)",
    )
    ap.add_argument(
        "--kappa_ext", default=2.6, type=float, help="Kappa value for transfer to extended threat model"
    )
    ap.add_argument(
        "--kappa_hyb", default=20.0, type=float, help="Kappa value for transfer back to hybrid threat model"
    )

    args = ap.parse_args()
    args.name = args.name[0]
    print(args)

    HYPER_PARAMS.training_episodes = args.epochs
    HYPER_PARAMS.scenario = args.scenario
    HYPER_PARAMS.num_nodes = args.num_nodes
    HYPER_PARAMS.red_add_flood_actions = args.red_add_flood_actions
    HYPER_PARAMS.red_add_stealth_action = args.red_add_stealth_action
    HYPER_PARAMS.red_sparse_reward = args.red_sparse_reward
    # def kappa_switch_condition_ext(red_progress_list):
    #     if len(red_progress_list) < 4:
    #         return False
    #     # Check if the average red progress reward over the last 4 iterations is less than or equal to kappa
    #     avg_red_progress = sum(red_progress_list) / 4
    #     return avg_red_progress <= args.kappa_max
    def kappa_switch_condition_extended(avg_red_progress_reward):
        return avg_red_progress_reward <= args.kappa_ext
    def kappa_switch_condition_hybrid(avg_red_progress_reward):
        return avg_red_progress_reward >= args.kappa_hyb
    HYPER_PARAMS.switch_condition_extended = kappa_switch_condition_extended
    HYPER_PARAMS.switch_condition_hybrid = kappa_switch_condition_hybrid
    
    
    HYPER_PARAMS.new_scenario = args.new_scenario
    HYPER_PARAMS.new_red_checkpoint = args.new_red_checkpoint
    HYPER_PARAMS.new_num_nodes = args.new_num_nodes
    
    red_agent = None
    if args.red_checkpoint:
        red_agent = load_red_inductive_ppo(args.red_checkpoint, globalnode=True, num_nodes=args.num_nodes)
    
    if args.use_blue_checkpoint:
        blue_agent = load_inductive_ppo(
            args.blue_checkpoint,
            globalnode=True,
            num_nodes=args.num_nodes,
        )
        blue_agent.train()
    elif args.inductive or args.attn or args.globalnode:
        blue_agent = InductiveGraphPPOAgent(
            DEFAULT_DIM,
            args.gnn,
            bs=HYPER_PARAMS.bs,
            a_kwargs={"lr": 0.0003, "hidden1": args.hidden, "hidden2": args.embedding, "num_nodes": args.num_nodes},
            c_kwargs={"lr": 0.001, "hidden1": args.hidden, "hidden2": args.embedding, "num_nodes": args.num_nodes},
            clip=0.2,
            epochs=HYPER_PARAMS.epochs,
            naive=not args.attn,
            globalnode=args.globalnode,
        )
    else:
        blue_agent = GraphPPOAgent(
            DEFAULT_DIM,
            bs=HYPER_PARAMS.bs,
            a_kwargs={"lr": 0.0003, "hidden1": args.hidden, "hidden2": args.embedding, "num_nodes": args.num_nodes},
            c_kwargs={"lr": 0.001, "hidden1": args.hidden, "hidden2": args.embedding, "num_nodes": args.num_nodes},
            clip=0.2,
            epochs=HYPER_PARAMS.epochs,
        )
    os.makedirs("checkpoints/blue", exist_ok=True)
    os.makedirs("logs/blue", exist_ok=True)
    os.makedirs(f"checkpoints/blue/{args.name}", exist_ok=True)
    HYPER_PARAMS.fnames = args.name
    train(blue_agent, red_agent, HYPER_PARAMS)
