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

def bi_graph(num_nodes):
    links=[]
    for i in range(num_nodes):
        l = [0] * num_nodes
        graph_idx= i % 2
        for j in range(num_nodes):
            graph_j_idx = j % 2
            if graph_idx != graph_j_idx:
                l[j] = 1
        links.append(l)
    return Topology(f'BiGraph(n={num_nodes})', links)

def alltoall_bigraph(num_ranks, instances, protocol):
    topology = bi_graph(num_ranks)
    micro_chunks = num_ranks // 2
    collective = AllToAll(num_ranks, micro_chunks, inplace=False)
    
    with MSCCLProgram("alltoall_bigraph", topology, collective, instances=instances, protocol=protocol, threadblock_policy=ThreadblockPolicy.manual):
        for rank in range(num_ranks):
            graph_idx = rank % 2
            for index in range(num_ranks):
                dst_rank = index
                if dst_rank == rank:
                    chunk(rank, Buffer.input, index * micro_chunks, micro_chunks).copy(rank, Buffer.output, index * micro_chunks)
                    continue

                dst_graph = dst_rank % 2
                if dst_graph != graph_idx:
                    chunk(rank, Buffer.input, index * micro_chunks, micro_chunks).copy(dst_rank, Buffer.output, rank * micro_chunks, sendtb=dst_rank, recvtb=rank)
                    continue
                
                ## same graph
                for micro_chunk_idx in range(micro_chunks):
                    int_graph_idx = (graph_idx + 1) % 2
                    int_rank = micro_chunk_idx * 2 + int_graph_idx
                    chunk(rank, Buffer.input, index * micro_chunks + micro_chunk_idx)\
                        .copy(int_rank, f'Scratch_{rank}_{index}', sendtb=int_rank, recvtb=rank)\
                        .copy(dst_rank, Buffer.output, rank * micro_chunks + micro_chunk_idx, sendtb=dst_rank, recvtb=int_rank)



        XML()
        Check()


parser = argparse.ArgumentParser()
parser.add_argument('num_gpus', type=int, help ='number of gpus')
parser.add_argument('instances', type=int, help ='number of instances')
parser.add_argument('--protocol', type=str, default='Simple', choices=['Simple', 'LL', 'LL128'], help ='NCCL protocol. Default: Simple')
args = parser.parse_args()

alltoall_bigraph(args.num_gpus, args.instances, args.protocol)
