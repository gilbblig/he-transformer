#!/bin/sh

python train.py
python -m tensorflow.python.tools.freeze_graph --input_graph graph.pb --input_checkpoint model --output_graph graph_frozen.pb --output_node_names=theend
python -m tensorflow.python.tools.optimize_for_inference --input graph_frozen.pb --output graph_optimized.pb --input_names=inx,iny --output_names=theend
NGRAPH_ENCRYPT_DATA=1 NGRAPH_HE_SEAL_CONFIG=../../test/model/experiment.json NGRAPH_TF_BACKEND=HE_SEAL python test.py --batch_size=4096
