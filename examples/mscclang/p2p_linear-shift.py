import argparse
from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllToAll

def alltoall_p2p_linear_shift(num_ranks, instances, gpus_per_node, protocol):
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)

    with MSCCLProgram("alltoall_p2p_linear-shift", topology, collective, instances=instances, \
        protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        num_nodes = num_ranks // gpus_per_node
        # --- Phase 1: (Intra-node) P2P ---
        for n in range(num_nodes):
            for g in range(gpus_per_node):
                src_rank = g + n * gpus_per_node
                for index in range(num_ranks):
                    dst_g = index % gpus_per_node
                    dst_rank = dst_g + n * gpus_per_node
                    index_n = index // gpus_per_node
                    index_g = index % gpus_per_node
                    dst_index_n = index_n
                    dst_index_g = g
                    dst_index = dst_index_n * gpus_per_node + dst_index_g
                    c = chunk(src_rank, Buffer.input, index)
                    c.copy(dst_rank, Buffer.scratch, dst_index, sendtb=dst_rank, recvtb=src_rank)
        
        # --- Phase 2: (Intra-node) Ring(Linear-shift step) ---
        for step in range(num_nodes):
            for n in range(num_nodes):
                for g in range(gpus_per_node):
                    src_rank = g + n * num_nodes
                    dst_g = g
                    dst_n = (n + step) % num_nodes
                    dst_rank = dst_g + dst_n * gpus_per_node

                    src_index_n = dst_n
                    src_index_begin = src_index_n * gpus_per_node
                    dst_index_n = n
                    dst_index_begin = dst_index_n * gpus_per_node
                    c = chunk(src_rank, Buffer.scratch, src_index_begin, gpus_per_node)
                    c = c.copy(src_rank, Buffer.scratch, num_ranks)
                    c.copy(dst_rank, Buffer.output, dst_index_begin, sendtb=dst_rank, recvtb=src_rank)
        XML()
        Check()

parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('gpus_per_node', type=int, help ='number of gpus per node')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

alltoall_p2p_linear_shift(args.num_gpus, args.instances, args.gpus_per_node, args.protocol)
