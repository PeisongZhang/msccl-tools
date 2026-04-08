import argparse

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllToAll

## chunks belong to rank.input:  [rank * chunks_per_rank, (rank + 1) * chunks_per_rank)
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

def alltoall_ring_ring(num_ranks, instances, gpus_per_row, protocol):
    assert num_ranks % gpus_per_row == 0, \
        "num_ranks 必须能被 gpus_per_row 整除，才能构成整齐的 2D 网格"

    # topology = fully_connected(num_ranks)
    topology = twod_torus(num_ranks)
    collective = AllToAll(num_ranks, 2, inplace=False)

    with MSCCLProgram("alltoall_ring_ring", topology, collective,
                      instances=instances, protocol=protocol):
        num_row = num_ranks // gpus_per_row
        num_column = gpus_per_row

        ## 先写一个完整环的all-to-all
        ## bidirectional
        ## step : [0, ring length]

        ## 现在要把完整环改成半环all-to-all
        ## bidirectional half ring
        ## step: [0, ring length / 2]

        ## chunks: num_ranks * 2
        ## chunks inplace to a rank: chunks // 2 == rank_id
        ## for step 0: copy 2 chunks from input to output for every rank
        ##     step i:  forward chunk_id: (rank_id + i) % num_ranks
        ##                      sub_chunk: chunk_id * 2  
        #                       sub_chunk + 1
        ##              backward chunk_id: (rank_id - i) % num_ranks
        ##                      sub_chunk: chunk_id * 2 + 1 
        #                       sub_chunk - 1
        # .             assert forward chunk id != backward chunk id
        ##     if forward chunk id == backward chunk id
        ##        forward sub_chunk: chunk_id * 2
        ##        backward sub_chunk: chunk_id * 2 + 1
        for row in range(num_row):
            for step in range(num_column // 2 + 1):
                for column in range(num_column):
                    rank = row * num_column + column
                    ## chunk_id = [(r * num_column + column) for r in range(num_row)]

                    if step == 0:
                        chunk_id = []
                        for r in range(num_row):
                            c_id = r * num_column + column
                            chunk_id.append(c_id * 2)
                            chunk_id.append(c_id * 2 + 1)
                        for id in chunk_id:
                            chunk(rank, Buffer.input, id).copy(rank, Buffer.scratch, id) 
                        continue 
                    assert step != 0

                    src_column = column ## rank_id within row
                    dst_column1 = (column + step) % num_column ## rank id within a row
                    dst_column2 = (column - step) % num_column ## rank id within a row

                    src_rank = row * num_column + src_column ## rank id
                    dst_rank1 = row * num_column + dst_column1
                    dst_rank2 = row * num_column + dst_column2
                    src_chunk_id1 = []
                    src_chunk_id2 = []
                    dst_chunk_id1 = []
                    dst_chunk_id2 = []

                    if dst_rank1 == dst_rank2: ## dst_column1 == dst_column2
                        src_chunk_id1 = [(r * num_column + dst_column1) * 2 for r in range(num_row)] # forward
                        src_chunk_id2 = [(r * num_column + dst_column2) * 2 + 1 for r in range(num_row)] # backward

                        dst_chunk_id1 = [(r * num_column + src_column) * 2 for r in range(num_row)] # forward
                        dst_chunk_id2 = [(r * num_column + src_column) * 2 + 1 for r in range(num_row)]
                    else:
                        ## forward
                        for r in range(num_row):
                            src_chunk_id1.append((r * num_column + dst_column1) * 2)
                            src_chunk_id1.append((r * num_column + dst_column1) * 2 + 1)

                            dst_chunk_id1.append((r * num_column + src_column) * 2)
                            dst_chunk_id1.append((r * num_column + src_column) * 2 + 1)
                        ## backward
                            src_chunk_id2.append((r * num_column + dst_column2) * 2)
                            src_chunk_id2.append((r * num_column + dst_column2) * 2 + 1)
                            dst_chunk_id2.append((r * num_column + src_column) * 2)
                            dst_chunk_id2.append((r * num_column + src_column) * 2 + 1)
                        


                    for current_step in range(step):
                        current_column1 = (column + current_step) % num_column
                        next_column1 = (current_column1 + 1) % num_column
                        current_rank1 = row * num_column + current_column1
                        next_rank1 = row * num_column + next_column1

                        if current_rank1 == src_rank: # substage 1
                            for i in range(len(src_chunk_id1)):
                                c = chunk(current_rank1, Buffer.input, src_chunk_id1[i])
                                if next_rank1 == dst_rank1:
                                    c.copy(dst_rank1, Buffer.scratch, dst_chunk_id1[i])
                                else:
                                    c.copy(next_rank1, Buffer.output, dst_chunk_id1[i])
                        else:
                            for i in range(len(src_chunk_id1)):
                                c = chunk(current_rank1, Buffer.output, dst_chunk_id1[i])
                                if next_rank1 == dst_rank1:
                                    c.copy(dst_rank1, Buffer.scratch, dst_chunk_id1[i])
                                else:
                                    c.copy(next_rank1, Buffer.output, dst_chunk_id1[i])

                        current_column2 = (column - current_step) % num_column
                        next_column2 = (current_column2 - 1) % num_column
                        current_rank2 = row * num_column + current_column2
                        next_rank2 = row * num_column + next_column2
                        if current_rank2 == src_rank:
                            for i in range(len(src_chunk_id2)):
                                c = chunk(current_rank2, Buffer.input, src_chunk_id2[i])
                                if next_rank2 == dst_rank2:
                                    c.copy(dst_rank2, Buffer.scratch, dst_chunk_id2[i])
                                else:
                                    c.copy(next_rank2, Buffer.output, dst_chunk_id2[i])
                        else:
                            for i in range(len(src_chunk_id2)):
                                c = chunk(current_rank2, Buffer.output, dst_chunk_id2[i])
                                if next_rank2 == dst_rank2:
                                    c.copy(dst_rank2, Buffer.scratch, dst_chunk_id2[i])
                                else:
                                    c.copy(next_rank2, Buffer.output, dst_chunk_id2[i])


        for column in range(num_column):  # 列内all-to-all
            for step in range(num_row // 2 + 1):  # 列内步数
                for row in range(num_row): # 列内每一行（每一个rank)
                    rank = row * num_column + column # rank_id
                    ## chunk_id, 数量和列数一致
                    # chunk_id = [(num_column * row + c) for c in range(num_column)]
                    if step == 0:
                        chunk_id = []
                        for c in range(num_column):
                            c_id = num_column * row + c
                            chunk_id.append(c_id * 2)
                            chunk_id.append(c_id * 2 + 1)
                        for id in chunk_id:
                            chunk(rank, Buffer.scratch, id).copy(rank, Buffer.output, id)
                        continue
                    assert step != 0

                    ## step != 0
                    src_row = row # rank id within a column
                    dst_row1 = (row + step) % num_row ## rank id within a row
                    dst_row2 = (row - step) % num_row

                    src_rank = src_row * num_column + column
                    dst_rank1 = dst_row1 * num_column + column
                    dst_rank2 = dst_row2 * num_column + column

                    # src_chunk_id1 = [(dst_row1 * num_column + c) * 2 for c in range(num_column)]
                    # src_chunk_id2 = [(dst_row2 * num_column + c) * 2 + 1 for c in range(num_column)]

                    # dst_chunk_id1 = [(src_row * num_column + c) * 2 for c in range(num_column)]
                    # dst_chunk_id2 = [(src_row * num_column + c) * 2 + 1 for c in range(num_column)]
                    src_chunk_id1 = []
                    src_chunk_id2 = []
                    dst_chunk_id1 = []
                    dst_chunk_id2 = []

                    if dst_rank1 == dst_rank2:
                        src_chunk_id1 = [(dst_row1 * num_column + c) * 2 for c in range(num_column)]
                        src_chunk_id2 = [(dst_row1 * num_column + c) * 2 + 1 for c in range(num_column)]
                        dst_chunk_id1 = [(src_row * num_column + c) * 2 for c in range(num_column)]
                        dst_chunk_id2 = [(src_row * num_column + c) * 2 + 1 for c in range(num_column)]
                    else:
                        for c in range(num_column):
                            src_chunk_id1.append((dst_row1 * num_column + c) * 2)
                            src_chunk_id1.append((dst_row1 * num_column + c) * 2 + 1)
                            dst_chunk_id1.append((src_row * num_column + c) * 2)
                            dst_chunk_id1.append((src_row * num_column + c) * 2 + 1)
                        
                            src_chunk_id2.append((dst_row2 * num_column + c) * 2)
                            src_chunk_id2.append((dst_row2 * num_column + c) * 2 + 1)
                            dst_chunk_id2.append((src_row * num_column + c) * 2)
                            dst_chunk_id2.append((src_row * num_column + c) * 2 + 1)


                    for current_step in range(step):
                        current_row1 = (row + current_step) % num_row
                        next_row1 = (current_row1 + 1) % num_row
                        current_rank1 = current_row1 * num_column + column
                        next_rank1 = next_row1 * num_column + column

                        if current_rank1 == src_rank: # substage 1
                            for i in range(len(src_chunk_id1)):
                                c = chunk(current_rank1, Buffer.scratch, src_chunk_id1[i])
                                if next_rank1 == dst_rank1:
                                    c.copy(dst_rank1, Buffer.output, dst_chunk_id1[i])
                                else:
                                    c.copy(next_rank1, Buffer.input, dst_chunk_id1[i])
                        else:
                            for i in range(len(src_chunk_id1)):
                                c = chunk(current_rank1, Buffer.input, dst_chunk_id1[i])
                                if next_rank1 == dst_rank1:
                                    c.copy(dst_rank1, Buffer.output, dst_chunk_id1[i])
                                else:
                                    c.copy(next_rank1, Buffer.input, dst_chunk_id1[i])

                        current_row2 = (row - current_step) % num_row
                        next_row2 = (current_row2 - 1) % num_row
                        current_rank2 = current_row2 * num_column + column
                        next_rank2 = next_row2 * num_column + column

                        if current_rank2 == src_rank:
                            for i in range(len(src_chunk_id2)):
                                c = chunk(current_rank2, Buffer.scratch, src_chunk_id2[i])
                                if next_rank2 == dst_rank2:
                                    c.copy(dst_rank2, Buffer.output, dst_chunk_id2[i])
                                else:
                                    c.copy(next_rank2, Buffer.input, dst_chunk_id2[i])
                        else:
                            for i in range(len(src_chunk_id2)):
                                c = chunk(current_rank2, Buffer.input, dst_chunk_id2[i])
                                if next_rank2 == dst_rank2:
                                    c.copy(dst_rank2, Buffer.output, dst_chunk_id2[i])
                                else:
                                    c.copy(next_rank2, Buffer.input, dst_chunk_id2[i])



        XML()
        Check()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('num_gpus', type=int, help='number of gpus')
    parser.add_argument('instances', type=int, help='number of instances')
    parser.add_argument('gpus_per_row', type=int, help='number of gpus per row')
    parser.add_argument(
        '--protocol',
        type=str,
        default='Simple',
        choices=['Simple', 'LL', 'LL128'],
        help='NCCL protocol. Default: Simple'
    )
    args = parser.parse_args()

    alltoall_ring_ring(
        args.num_gpus,
        args.instances,
        args.gpus_per_row,
        args.protocol
    )
