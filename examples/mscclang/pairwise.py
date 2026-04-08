import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllToAll

def alltoall_pairwise(num_ranks, instances, protocol):
    topology = fully_connected(num_ranks)
    collective = AllToAll(num_ranks, 1, inplace=False)

    with MSCCLProgram("alltoall_pairwise", topology, collective, instances=instances, protocol=protocol, threadblock_policy=ThreadblockPolicy.manual): 
        for rank in range(num_ranks):
            chunk(rank, Buffer.input, rank).copy(rank, Buffer.output, rank)

        for step in range(1, num_ranks):
            for rank in range(num_ranks):
                src_rank = rank
                dst_rank = rank ^ step
                src_chunk_id = dst_rank
                c = chunk(src_rank, Buffer.input, src_chunk_id)
                # c = c.copy(src_rank, Buffer.scratch, 0)
                c.copy(dst_rank, Buffer.output, src_rank, sendtb=dst_rank, recvtb=src_rank+num_ranks, ch=dst_rank)

        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

alltoall_pairwise(args.num_gpus, args.instances, args.protocol)
