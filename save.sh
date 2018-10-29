python3 freeze_graph.py --input_saved_model_dir $1 --output_node_names "Softmax" --saved_model_tags serve --output_graph $2
