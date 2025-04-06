# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Example of a simple custom collective where Rank 0 sends a chunk to Ranks 1 and 2

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import Collective
import argparse

# This class defines the requirements of a Broadcast collective, i.e. the end state after the collective. 
# Several implementations/algorithms of the collective could exist.
class Broadcast(Collective):
    # Initial state is chunk0 is on rank0 in the input buffer
    def init_buffers(self):
        if self.chunk_factor != 1:
            print(f'Chunk factor is {self.chunk_factor} should be 1')
            exit()

        rank_buffers = []
        for r in range(self.num_ranks):
            input_buffer = [None]
            output_buffer = [None]
            if r == 0:
                # Format for specifying a chunk
                # Chunk(starting rank, starting index, ending rank, ending index)
                # Because this chunk ends up on multiple ranks ending rank is set to -1
                input_buffer[0] = Chunk(r, 0, -1, 0)
            buffers = {Buffer.input : input_buffer, 
                       Buffer.output : output_buffer}
            rank_buffers.append(buffers)
        return rank_buffers
            

    # Final state chunk0 from rank0 is in the output buffer of rank1 and rank2
    def check(self, prog):
        correct = True
        for r in range(1, self.num_ranks):
            output = prog.buffers[r][Buffer.output]
            for c in range(self.chunk_factor):
                chunk = output[c]
                # Check that we got chunk 0 from rank 0
                if chunk is None or chunk.origin_rank != 0 or chunk.origin_index != 0:
                    print(f'Rank {r} chunk {c} is incorrect should be ({0}, {0}) given {chunk}')
                    correct = False
        r = 0
        output = prog.buffers[r][Buffer.output]
        for c in range(self.chunk_factor):
            chunk = output[c]
            if chunk is not None:
                print(f'Rank 0\'s output is not None')
                correct = False
        if self.chunk_factor != 1:
            print(f'chunk_factor is {self.chunk_factor} and not 1')
            correct = False
        return correct


def broadcast_simple(size):
    # A simple broadcast operation that sequentially broacasts from root 0
    topology = fully_connected(size) 
    # Collectives take in number of ranks in the network, chunksperloop of the collective, whether it is inplace, 
    collective = Broadcast(size, 1, inplace=False)
    with MSCCLProgram("broadcast_simple", topology, collective, instances=1, protocol="Simple"):
        # Get the chunk at rank 0 index 0 of the input buffer
        c = chunk(0, Buffer.input, 0)
        # Send chunks to the other nodes 
        # Can specify the sender's tb, receiver's tb, and channel for the send operation
        # MSCCLang provides a default threadblock assignment if they aren't specified
        # MSCCLang will also check the tb/channel combos are valid
        for dst_rank in range(1, size):
            c.copy(dst_rank, buffer=Buffer.output, index=0, sendtb=1, recvtb=1, ch=0)

        XML() # Generates the XML for this collective
        Check() # Checks the routes defined for each chunk are correct. Currently doesn't check XML correct



parser = argparse.ArgumentParser(description="Run the broadcast_simple example.")
parser.add_argument("--num_ranks", type=int, default=4, help="The size of the network (must be at least 2). Default is 4.")
args = parser.parse_args()

broadcast_simple(args.num_ranks)
