import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllToAll

def twod_torus(num_nodes):
    links = []
    for i in range(num_nodes):
        row = i // 4
        col = i % 4
        l = [0] * num_nodes
        a = (i + 4) % num_nodes
        b = (i - 4) % num_nodes
        c =  (row * 4) + (col - 1) % 4
        d =  (row * 4) + (col + 1) % 4
        l[(i + 4) % num_nodes] = 1
        l[(i - 4) % num_nodes] = 1
        l[(row * 4) + (col - 1) % 4] = 1
        l[(row * 4) + (col + 1) % 4] = 1
        links.append(l)
    return Topology(f'2DTorus(n={num_nodes})', links)

def twod_fullmesh(num_nodes):
    # Assuming square grid (num_nodes should be a perfect square)
    # Calculate grid dimensions
    grid_size = int(num_nodes ** 0.5)
    if grid_size * grid_size != num_nodes:
        raise ValueError(f"num_nodes ({num_nodes}) must be a perfect square for 2D fullmesh topology")
    
    links = []
    for i in range(num_nodes):
        row = i // grid_size
        col = i % grid_size
        l = [0] * num_nodes
        
        # Connect to all nodes in the same row (full mesh within row)
        for j in range(grid_size):
            node_in_row = row * grid_size + j
            if node_in_row != i:
                l[node_in_row] = 1
        
        # Connect to all nodes in the same column (full mesh within column)
        for j in range(grid_size):
            node_in_col = j * grid_size + col
            if node_in_col != i:
                l[node_in_col] = 1
        
        links.append(l)
    return Topology(f'2DFullMesh(n={num_nodes})', links)


def alltoall_2dfullmesh(num_ranks, instances, protocol):
    topology = twod_fullmesh(num_ranks)
    collective = AllToAll(num_ranks, 2, inplace=False)
    
    # Calculate grid dimensions
    grid_size = int(num_ranks ** 0.5)
    num_rows = grid_size
    num_cols = grid_size

    with MSCCLProgram("alltoall_2dfullmesh", topology, collective, instances=instances, protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        for rank in range(num_ranks):
            row = rank // num_cols
            col = rank % num_cols
            
            for index in range(num_ranks):
                dst_rank = index
                dst_index = rank
                dst_rank_row = dst_rank // num_cols
                dst_rank_col = dst_rank % num_cols
                if dst_rank_row == row or dst_rank_col == col:
                    chunk(rank, Buffer.input, index * 2, 2).copy(dst_rank, Buffer.output, dst_index * 2, sendtb=dst_rank, recvtb=rank)
                else:
                    int_row_rank = row * num_cols + dst_rank_col
                    chunk(rank, Buffer.input, index * 2).copy(int_row_rank, f'Scratch{rank}_{index}', sendtb=int_row_rank, recvtb=rank) \
                        .copy(dst_rank, Buffer.output, dst_index * 2, sendtb=dst_rank, recvtb=int_row_rank)
                    int_col_rank = dst_rank_row * num_cols + col
                    chunk(rank, Buffer.input, index * 2 + 1).copy(int_col_rank, f'Scratch{rank}_{index}', sendtb=int_col_rank, recvtb=rank) \
                        .copy(dst_rank, Buffer.output, dst_index * 2 + 1, sendtb=dst_rank, recvtb=int_col_rank)

        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

alltoall_2dfullmesh(args.num_gpus, args.instances, args.protocol)
