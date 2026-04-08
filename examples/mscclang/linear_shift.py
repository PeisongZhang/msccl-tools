import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllToAll

def alltoall_pairwise(num_ranks, instances, protocol):
    topology = fully_connected(num_ranks)
    chunks_per_rank = 1
    collective = AllToAll(num_ranks, chunks_per_rank, inplace=False)

    with MSCCLProgram("alltoall_pairwise", topology, collective, instances=instances, protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        for rank in range(num_ranks):
            chunk(rank, Buffer.input, rank).copy(rank, Buffer.output, rank)

        for step in range(1, num_ranks):
            for src_rank in range(num_ranks):
                dst_rank = (src_rank + step) % num_ranks
                c = chunk(src_rank, Buffer.input, dst_rank)
                c.copy(dst_rank, Buffer.output, src_rank, sendtb=dst_rank, recvtb=src_rank+num_ranks, ch=dst_rank)
        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

alltoall_pairwise(args.num_gpus, args.instances, args.protocol)
