for i in 0 1 2 3
do
    # RBT reward
    python red_ppo_train.py cardiff_red_gcn_A_rbt_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500
    python red_ppo_train.py cardiff_red_gcn_B_rbt_stealth_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --add_stealth_action
    python red_ppo_train.py cardiff_red_gcn_B_rbt_flood_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --add_flood_actions --flood_k 3
    python red_ppo_train.py cardiff_red_gcn_B_rbt_stealth_flood_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --add_flood_actions --flood_k 3 --add_stealth_action

    # RBT reward w/o BT policy conditioning
    python red_ppo_train.py cardiff_red_gcn_A_rbt_no_phase_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --exclude_bt_phase
    python red_ppo_train.py cardiff_red_gcn_B_rbt_no_phase_stealth_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --add_stealth_action --exclude_bt_phase
    python red_ppo_train.py cardiff_red_gcn_B_rbt_no_phase_flood_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --add_flood_actions --flood_k 3 --exclude_bt_phase
    python red_ppo_train.py cardiff_red_gcn_B_rbt_no_phase_stealth_flood_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --add_flood_actions --flood_k 3 --add_stealth_action --exclude_bt_phase

    # Task reward
    python red_ppo_train.py cardiff_red_gcn_A_base_reward_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --cc2_reward
    python red_ppo_train.py cardiff_red_gcn_B_base_reward_stealth_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --add_stealth_action --cc2_reward
    python red_ppo_train.py cardiff_red_gcn_B_base_reward_flood_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --add_flood_actions --flood_k 3 --cc2_reward
    python red_ppo_train.py cardiff_red_gcn_B_base_reward_stealth_flood_$i -s scenarios/cage2.yaml --num_nodes 13 --gnn gcn --cardiff --epochs 500 --add_flood_actions --flood_k 3 --add_stealth_action --cc2_reward
done

