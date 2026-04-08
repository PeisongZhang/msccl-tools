#!/bin/bash
python ./fullmesh.py 16 16 > p2p.xml  
python ./pairwise.py 16 16 > pairwise.xml 
python ./linear_shift.py 16 16 > linear_shift.xml 
python ./p2p_p2p.py 16 16 4 > p2p_p2p.xml 
python ./p2p_linear-shift.py 16 16 4 > p2p_linear-shift.xml
python ./p2p_pairwise.py 16 16 4 > p2p_pairwise.xml  
python ./bi_halfring_2d.py 16 16 4 > dor.xml
python ./bi_halfring_rotate.py 16 16 4 > drr.xml
python ./bi_graph.py 16 2 > bi_graph.xml
python ./2d_fullmesh.py 16 16 > 2d_fullmesh.xml

python add_pairwise_deps.py pairwise.xml pairwise_deps.xml 
cp pairwise_deps.xml pairwise.xml

# ython add_linear_shift_deps.py linear_shift.xml linear_shift_deps.xml
python add_linear_shift_deps.py --input linear_shift.xml --output linear_shift_deps.xml
cp linear_shift_deps.xml linear_shift.xml