python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --red_checkpoint checkpoints/red/red_gcn_B_rbt_stealth_flood_2.pt --add_flood_actions --add_stealth_action --blue_checkpoint checkpoints/blue/blue_gcn_B_rbt_stealth_flood_2.pt --fname blue_keep_rbt_stealth_flood
python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --red_checkpoint checkpoints/red/red_gcn_B_rbt_flood_0.pt --add_flood_actions --blue_checkpoint checkpoints/blue/blue_gcn_B_rbt_flood_0.pt --fname blue_gcn_B_rbt_flood_0
python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --red_checkpoint checkpoints/red/red_gcn_B_rbt_stealth_2.pt --add_stealth_action --blue_checkpoint checkpoints/blue/blue_gcn_B_rbt_stealth_2.pt --fname blue_gcn_B_rbt_stealth_2
python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --red_checkpoint checkpoints/red/red_gcn_A_rbt_1.pt --blue_checkpoint checkpoints/blue/blue_gcn_A_rbt_1.pt --fname blue_gcn_A_rbt_1

python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --blue_checkpoint checkpoints/blue/blue_gcn_B_rbt_stealth_flood_2.pt
python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --blue_checkpoint checkpoints/blue/blue_gcn_B_rbt_flood_0.pt
python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --blue_checkpoint checkpoints/blue/blue_gcn_B_rbt_stealth_2.pt
python evaluation.py -s scenarios/cage2.yaml --num_nodes 13 --blue_checkpoint checkpoints/blue/blue_gcn_A_rbt_1.pt