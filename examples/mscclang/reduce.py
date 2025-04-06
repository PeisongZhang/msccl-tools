# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Example of a simple custom collective where Rank 0 sends a chunk to Ranks 1 and 2

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import Collective
import argparse

# This class defines the requirements of a Reduce collective, i.e. the end state after the collective. 
# Several implementations/algorithms of the collective could exist.
class Reduce(Collective):
    # Initial state is chunk0 is on rank0 in the input buffer
    def init_buffers(self):
        if self.chunk_factor != 1:
            print(f'Chunk factor is {self.chunk_factor} should be 1')
            exit()
        
        rank_buffers = []
        r = 0
        input_buffer = [None]
        output_buffer = [None] * self.num_ranks
        buffers = {Buffer.input : input_buffer, 
                    Buffer.output : output_buffer}
        rank_buffers.append(buffers)
        
        for r in range(1, self.num_ranks):
            input_buffer = [None]
            output_buffer = [None] 
            # Format for specifying a chunk
            # Chunk(starting rank, starting index, ending rank, ending index)
            # Because this chunk ends up on multiple ranks ending rank is set to -1
            input_buffer[0] = Chunk(r, 0, 0, 0)
            buffers = {Buffer.input : input_buffer, 
                       Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers
            

    # Final state chunk0 from rank0 is in the output buffer of rank1 and rank2
    def check(self, prog):
        correct = True
        output = prog.buffers[0][Buffer.output]
        for r in range(1, self.num_ranks):
            chunk = output[r]
            if chunk is None or chunk.origin_rank != r or chunk.origin_index != 0:
                print(f'Given {chunk}, should be from {r}, {0} but is from ')
                correct = False

        for r in range(1, self.num_ranks):
            output = prog.buffers[r][Buffer.output]
            for c in range(self.chunk_factor):
                chunk = output[c]
                if chunk is not None:
                    print(f"Rank {r} chunk {c} is not None")
                    correct = False
        if self.chunk_factor != 1:
            print(f'chunk_factor is {self.chunk_factor} and not 1')
            correct = False
        return correct


def reduce_simple(size):
    # A simple reduce operation that sequentially broacasts from root 0
    topology = fully_connected(size) 
    # Collectives take in number of ranks in the network, chunksperloop of the collective, whether it is inplace, 
    collective = Reduce(size, 1, inplace=False)
    with MSCCLProgram("reduce_simple", topology, collective, instances=1, protocol="Simple"):
        # Get the chunk at rank 0 index 0 of the input buffer
        # Send chunks to the other nodes 
        # Can specify the sender's tb, receiver's tb, and channel for the send operation
        # MSCCLang provides a default threadblock assignment if they aren't specified
        # MSCCLang will also check the tb/channel combos are valid
        for src_rank in range(1, size):
            c = chunk(src_rank, Buffer.input, 0)
            c.copy(0, buffer=Buffer.output, index=src_rank, sendtb=1, recvtb=1, ch=0)

        XML() # Generates the XML for this collective
        Check() # Checks the routes defined for each chunk are correct. Currently doesn't check XML correct



parser = argparse.ArgumentParser(description="Run the reduce_simple example.")
parser.add_argument("--num_ranks", type=int, default=4, help="The size of the network (must be at least 2). Default is 4.")
args = parser.parse_args()

reduce_simple(args.num_ranks)
