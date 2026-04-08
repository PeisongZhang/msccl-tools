# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from msccl.language import *
from msccl.topologies import *
from msccl.language.collectives import AllGather

def allgather_direct_impl(size):
    # Loop over each chunk's root
    for r in range(size):
        # Get the chunk at rank r, input[r]
        c = chunk(r, Buffer.input, 0)

        # Copy chunk to the output buffer of every rank
        for dst in range(size):
            c = c.copy(dst, Buffer.output, r, sendtb=dst, recvtb=(size + r))


def allgather_direct(size):
    topology = fully_connected(size)
    collective = AllGather(size, 1, False)
    with MSCCLProgram("allgather_direct", topology, collective, 1, 'Simple', ThreadblockPolicy.manual):
        allgather_direct_impl(size)
        XML()
        Check()

allgather_direct(16)