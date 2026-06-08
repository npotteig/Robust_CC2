from argparse import ArgumentParser
from types import SimpleNamespace
from typing import Union

from joblib import Parallel, delayed
import torch

from CardiffUni.Agents.MainAgent import MainAgent
from CybORG import CybORG
from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent

from models.ppo import GraphPPOAgent, PPOMemory
from models.red_inductive_ppo import RedInductiveGraphPPOAgent
from models.inductive_ppo import load_inductive_ppo

from wrapper.red_observation_graph import HostType, CompromiseLevel

from adapters.RedGraphAdapter import RedGraphAdapter
from adapters.BlueGraphAdapter import BlueGraphAdapter
from adapters.BlueChallengeAdapter import BlueChallengeAdapter
from wrapper.red_blue_wrapper import RedBlueWrapper

from rewards.cc2_subtask_functions import (
    subtask_1_complete,
    subtask_1_port_connected,
    subtask_2_complete,
    subtask_2_port_connected,
    flood_check,
    impact_complete
)
from rewards.bt import BehaviourTreeConfig

N_JOBS = 25
N_THREADS = 16

torch.set_num_threads(N_THREADS)
HYPER_PARAMS = SimpleNamespace(
    N=25,  # How many episodes before training
    bs=625,  # How many steps to learn from at a time
    episode_len=100,
    training_episodes=10_001,
    epochs=4,
    reward_mode="bt_as_reward",
    bt_config=BehaviourTreeConfig(
        subtask_names=["To_Enterprise", "To_Op_Server"],
        subtask_complete_functions=[subtask_1_complete, subtask_2_complete],
        subtask_port_functions=[subtask_1_port_connected, subtask_2_port_connected],
        subtask_flood_function=flood_check,
        mission_complete_function=impact_complete,
        use_memory=False,
    ),
    exclude_bt_phase=False,
)
GENERATE_VIZ_DATA = False


@torch.no_grad()
def generate_episode(
    red_agent: GraphPPOAgent,
    blue_agent: GraphPPOAgent,
    hp,
    path,
    max_nodes,
    add_flood_actions=False,
    add_stealth_action=False,
    use_cardiff_adapter=False,
):
    torch.set_num_threads(1)

    cyborg = CybORG(path, "parallel_sim")
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
        max_nodes=max_nodes,
        bt_config=hp.bt_config,
        reward_mode=hp.reward_mode,
        add_flood_actions=add_flood_actions,
        add_stealth_action=add_stealth_action,
        use_bt_phase=not hp.exclude_bt_phase,
    )
    wrapped = RedBlueWrapper(cyborg, red_adapter, blue_adapter)

    mem = PPOMemory(0)
    flooded_hosts_lengths = []
    impact_counts = 0
    state, info = wrapped.reset()

    for step_cnt in range(hp.episode_len):
        blue_action = blue_agent.get_action(state["Blue"])
        red_action, value, prob = red_agent.get_action(state["Red"])
        next_state, reward, done, truncated, info = wrapped.step(
            red_action_int=red_action, blue_action_int=blue_action
        )
        mem.remember(
            state["Red"],
            red_action,
            value,
            prob,
            reward["Red"],
            0 if step_cnt < hp.episode_len - 1 else 1,
        )
        flooded_hosts_lengths.append(len(wrapped.red_adapter.graph.flooded_nodes))
        impact_counts += int(info["Red"]["success"])
        state = next_state

    return (
        mem,
        impact_counts,
        max(flooded_hosts_lengths) if flooded_hosts_lengths else 0,
    )


def train(
    red_agent: GraphPPOAgent,
    blue_agent: Union[GraphPPOAgent, type[BaseAgent]],
    hp,
    path,
    max_nodes,
    add_flood_actions=False,
    add_stealth_action=False,
    use_cardiff_adapter=False,
):
    red_agent.train()
    if isinstance(blue_agent, GraphPPOAgent):
        blue_agent.eval()

    with open(f"logs/red/{hp.fnames}.txt", "w+") as f:
        f.write("epoch,avg_r,loss,avg_impact_actions,max_flooded\n")

    for e in range(1, hp.training_episodes):
        ret = Parallel(n_jobs=N_JOBS, prefer="processes")(
            delayed(generate_episode)(
                red_agent,
                blue_agent,
                hp,
                path,
                max_nodes,
                add_flood_actions,
                add_stealth_action,
                use_cardiff_adapter,
            )
            for _ in range(hp.N)
        )
        mems, impact_counts, flooded_host_lengths = zip(*ret)
        mem = PPOMemory(hp.bs).load(mems)
        red_agent.memory = mem
        avg_r = sum(mem.r) / hp.N
        avg_flooded = sum(flooded_host_lengths) / hp.N
        avg_impact_actions = sum(impact_counts) / hp.N

        torch.set_num_threads(N_THREADS)
        print(
            f"[{e}] Average reward: {avg_r}, Average impact actions: {avg_impact_actions}, Average flooded hosts: {avg_flooded}"
        )
        loss = red_agent.learn(verbose=False)

        with open(f"logs/red/{hp.fnames}.txt", "a") as f:
            f.write(f"{e},{avg_r},{loss},{avg_impact_actions},{avg_flooded}\n")

    red_agent.save(outf=f"checkpoints/red/{hp.fnames}.pt")


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
    ap.add_argument("--num_nodes", default=13, type=int)
    ap.add_argument(
        "--globalnode",
        action="store_true",
        help="Use KEEP",
    )
    ap.add_argument(
        "--cardiff",
        action="store_true",
        help="Use CardiffUni"
    )
    ap.add_argument(
        "--hidden",
        action="store",
        type=int,
        default=64,
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
    ap.add_argument("--epochs", default=1000, type=int)
    ap.add_argument(
        "--add_flood_actions",
        action="store_true",
        help="Whether to add flood actions to the action space",
    )
    ap.add_argument(
        "--add_stealth_action",
        action="store_true",
        help="Whether to add a stealth action to the action space",
    )
    ap.add_argument(
        "--flood_k", default=6, type=int, help="Number of hosts red needs to flood"
    )
    ap.add_argument(
        "--cc2_reward",
        action="store_true",
        help="Whether to only give rewards for successful impact of the operational servers",
    )
    ap.add_argument(
        "--exclude_bt_phase",
        action="store_true",
        help="Whether to exclude the behaviour tree phase from the observation and reward calculation",
    )

    args = ap.parse_args()
    args.name = args.name[0]
    path = args.scenario
    num_nodes = args.num_nodes

    # make checkpoints directory if it doesn't exist
    import os

    if not os.path.exists("checkpoints/red"):
        os.makedirs("checkpoints/red")
    
    if not os.path.exists("logs/red"):
        os.makedirs("logs/red")

    HYPER_PARAMS.training_episodes = args.epochs
    if args.cc2_reward:
        HYPER_PARAMS.reward_mode = "environment"
    HYPER_PARAMS.exclude_bt_phase = args.exclude_bt_phase

    num_bits = (num_nodes).bit_length()
    # num_bits = 0
    if args.add_flood_actions:
        default_dim = num_bits + HostType.__len__() + CompromiseLevel.__len__() + 1 + 1
        action_space = 4
        HYPER_PARAMS.bt_config.flood_k = args.flood_k
        HYPER_PARAMS.bt_config.flood_node_ids = range(num_nodes)
    else:
        default_dim = num_bits + HostType.__len__() + CompromiseLevel.__len__() + 1
        action_space = 2
    red_agent = RedInductiveGraphPPOAgent(
        default_dim,
        args.gnn,
        bs=HYPER_PARAMS.bs,
        a_kwargs={
            "lr": 0.0003,
            "hidden1": args.hidden,
            "hidden2": args.embedding,
            "num_nodes": num_nodes,
            "action_space": action_space,
        },
        c_kwargs={
            "lr": 0.001,
            "hidden1": args.hidden,
            "hidden2": args.embedding,
            "num_nodes": num_nodes,
        },
        clip=0.2,
        epochs=HYPER_PARAMS.epochs,
        globalnode=True,
    )

    # Load blue agent
    if args.cardiff:
        blue_agent = MainAgent()
    else:
        blue_agent = load_inductive_ppo(
            "model_weights/inductive_global-attn.pt", globalnode=True, num_nodes=num_nodes
        )
        blue_agent.set_deterministic(True)

    HYPER_PARAMS.fnames = args.name

    # generate_episode(red_agent, blue_agent, HYPER_PARAMS, path, num_nodes, add_flood_actions=args.add_flood_actions, add_stealth_action=args.add_stealth_action)
    train(
        red_agent,
        blue_agent,
        HYPER_PARAMS,
        path,
        num_nodes,
        add_flood_actions=args.add_flood_actions,
        add_stealth_action=args.add_stealth_action,
        use_cardiff_adapter=args.cardiff,
    )
