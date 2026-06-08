# BLine & Meander
python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode


# for i in 0 1 2 3
do
    # RBT reward
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_B_rbt_stealth_flood_$i.pt --add_stealth_action --add_flood_actions --fname keep_rbt_stealth_flood_$i
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_B_rbt_stealth_$i.pt --add_stealth_action --fname keep_rbt_stealth_$i
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_B_rbt_flood_$i.pt --add_flood_actions --fname keep_rbt_flood_$i
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_A_rbt_$i.pt --fname keep_rbt_$i

    # Task reward
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_B_base_reward_stealth_flood_$i.pt --add_stealth_action --add_flood_actions --cc2_reward --fname keep_base_reward_stealth_flood_$i
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_B_base_reward_stealth_$i.pt --add_stealth_action --cc2_reward --fname keep_base_reward_stealth_$i
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_B_base_reward_flood_$i.pt --add_flood_actions --cc2_reward --fname keep_base_reward_flood_$i
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_A_base_reward_$i.pt --cc2_reward --fname keep_base_reward_$i

    # RBT reward w/o BT policy conditioning
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_B_rbt_no_phase_stealth_flood_$i.pt --add_stealth_action --add_flood_actions --exclude_bt_phase --fname keep_rbt_no_phase_stealth_flood_$i
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_B_rbt_no_phase_stealth_$i.pt --add_stealth_action --exclude_bt_phase --fname keep_rbt_no_phase_stealth_$i
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_B_rbt_no_phase_flood_$i.pt --add_flood_actions --exclude_bt_phase --fname keep_rbt_no_phase_flood_$i
    python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --globalnode --red_checkpoint checkpoints/red/red_gcn_A_rbt_no_phase_$i.pt --exclude_bt_phase --fname keep_rbt_no_phase_$i
done