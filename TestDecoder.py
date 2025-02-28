import csv
import struct
import sys
import math
import matplotlib.pyplot as plt

def as_float(x):
    """Reinterpret a 32-bit unsigned int as a float."""
    return struct.unpack('f', struct.pack('I', x))[0]

def ByteAlign(numBits):
    """Align the number of bits to the next multiple of 8."""
    return ((numBits + 7) // 8) * 8

def firstbithigh(x):
    """Return the index (0-based) of the highest set bit.
       For x==0, returns 0."""
    return x.bit_length() - 1 if x != 0 else 0

def to_signed(x):
    """Convert a 32-bit unsigned int to a signed integer."""
    if x & 0x80000000:
        return -((~x & 0xFFFFFFFF) + 1)
    return x

def align(a, b, bitPos):
    """
    Align bits from two consecutive dwords 'a' and 'b' given the absolute bit position.
    The bit offset within the first dword is (bitPos % 32).
    """
    off = bitPos % 32
    if off == 0:
        return a
    return ((a >> off) | (b << (32 - off))) & 0xFFFFFFFF

class DGFHeader:
    def __init__(self):
        self.bitsPerComponent = (0, 0, 0)  # (x, y, z)
        self.numTriangles = 0
        self.numVerts = 0
        self.bitsPerIndex = 0
        self.anchor = (0, 0, 0)            # (x, y, z)
        self.scale = 0.0
        self.primIDBase = 0
        self.userData = 0
        self.bitSize = 0
        self.geomIDMeta = 0
        self.haveGeomIDPalette = False

    def __str__(self):
        return (f"DGFHeader(numTriangles={self.numTriangles}, numVerts={self.numVerts}, "
                f"bitsPerIndex={self.bitsPerIndex}, bitsPerComponent={self.bitsPerComponent}, "
                f"anchor={self.anchor}, scale={self.scale}, primIDBase={self.primIDBase}, "
                f"userData={self.userData}, bitSize={self.bitSize}, "
                f"geomIDMeta={self.geomIDMeta}, haveGeomIDPalette={self.haveGeomIDPalette})")

class DGFBlockInfo:
    def __init__(self):
        self.header = None
        self.dgfBuffer = None
        self.blockStartOffset = 0
        self.bitsPerVertex = 0
        self.vertexBitStart = 0
        self.geomIDBitStart = 0
        self.indexBitStart = 0

    def __str__(self):
        return (f"DGFBlockInfo(blockStartOffset={self.blockStartOffset}, bitsPerVertex={self.bitsPerVertex}, "
                f"vertexBitStart={self.vertexBitStart}, geomIDBitStart={self.geomIDBitStart}, "
                f"indexBitStart={self.indexBitStart}, header={self.header})")

class ByteAddressBuffer:
    def __init__(self, data):
        # data is a flat list of 32-bit unsigned ints.
        self.data = data

    def Load4(self, byte_offset):
        """Load 4 consecutive 32-bit words starting at byte_offset."""
        index = byte_offset // 4
        return self.data[index:index+4]

    def Load2(self, byte_offset):
        """Load 2 consecutive 32-bit words starting at byte_offset."""
        index = byte_offset // 4
        return self.data[index:index+2]

    def Load3(self, byte_offset):
        """Load 3 consecutive 32-bit words starting at byte_offset."""
        index = byte_offset // 4
        return self.data[index:index+3]

    def Load(self, byte_offset):
        """Load a single 32-bit word from the given byte offset."""
        index = byte_offset // 4
        return self.data[index]

def DGFLoadHeader(dgfBuffer, blockStartOffset):
    header = DGFHeader()
    # Load 4 uints (16 bytes) from the buffer.
    H = dgfBuffer.Load4(blockStartOffset)
    h0, h1, h2, h3 = H[0], H[1], H[2], H[3]
    # Load 2 uints (8 bytes) from blockStartOffset + 16.
    H2 = dgfBuffer.Load2(blockStartOffset + 16)
    h4, h5 = H2[0], H2[1]
    
    header.numTriangles = ((h0 >> 16) & 0x3f) + 1
    header.numVerts     = ((h0 >> 10) & 0x3f) + 1
    header.bitsPerIndex = ((h0 >> 8)  & 3) + 3

    bpc_x = (h2 & 0xf) + 1
    bpc_y = ((h2 >> 4) & 0xf) + 1
    bpc_z = (h3 & 0xf) + 1
    header.bitsPerComponent = (bpc_x, bpc_y, bpc_z)

    anchor_x = to_signed(h1) >> 8
    anchor_y = to_signed(h2) >> 8
    anchor_z = to_signed(h3) >> 8
    header.anchor = (anchor_x, anchor_y, anchor_z)

    header.scale = as_float((h1 & 0xff) << 23)

    header.primIDBase = h4 & ((1 << 29) - 1)
    haveUserData = ((h4 >> 29) & 1)
    header.userData = haveUserData * h5
    header.bitSize = 32 * (5 + haveUserData)
    header.haveGeomIDPalette = ((h3 & 0x80) != 0)
    header.geomIDMeta = h0 >> 22

    return header

def ComputeGeomIDPaletteSize(header):
    geomIDMeta    = header.geomIDMeta
    numGeomIDs    = (geomIDMeta >> 5) + 1
    prefixBitSize = (geomIDMeta & 0x1F)
    payloadBitSize = 25 - prefixBitSize
    indexBitSize  = firstbithigh(numGeomIDs - 1) + 1
    totalBits     = numGeomIDs * payloadBitSize + header.numTriangles * indexBitSize + prefixBitSize
    paletteSize   = ByteAlign(totalBits)
    return paletteSize if header.haveGeomIDPalette else 0

def DGFLoadBlockInfo(dgfBuffer, dgfBlockIndex):
    blockInfo = DGFBlockInfo()
    # Each block is 128 bytes.
    blockInfo.blockStartOffset = dgfBlockIndex * 128
    blockInfo.dgfBuffer = dgfBuffer
    blockInfo.header = DGFLoadHeader(dgfBuffer, blockInfo.blockStartOffset)
    blockInfo.bitsPerVertex = sum(blockInfo.header.bitsPerComponent)
    
    vertexBitSize = ByteAlign(blockInfo.header.numVerts * blockInfo.bitsPerVertex)
    geomIDPaletteSize = ComputeGeomIDPaletteSize(blockInfo.header)
    
    blockInfo.vertexBitStart = blockInfo.header.bitSize
    blockInfo.geomIDBitStart = blockInfo.vertexBitStart + vertexBitSize
    blockInfo.indexBitStart  = blockInfo.geomIDBitStart + geomIDPaletteSize
    return blockInfo

def load_csv_as_uints(csv_filename):
    """Load the CSV file and extract the 'data' column as a flat list of integers."""
    data = []
    with open(csv_filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 2:
                continue
            cell = row[1].strip()
            if cell:
                data.append(int(cell))
    return data

def LoadTriangleControlValues(s, triangleId):
    """
    Decode the 2-bit control value for a given triangle in block s.
    
    If triangleId == 0, return the restart control code.
    Otherwise, adjust the triangleId (storedTriangleId = triangleId - 1) and:
      - Compute the dword offset where the control values are stored.
      - Extract the proper 2-bit code (the codes are stored in reverse order in each dword).
    """
    DGF_CTRL_RESTART = 0  # Define the restart control code.
    if triangleId == 0:
        return DGF_CTRL_RESTART

    storedTriangleId = triangleId - 1
    # Control bytes start at 28 * 4 bytes into the block.
    # There are 16 codes per dword and the dwords are stored in reverse order.
    dword_offset = s.blockStartOffset + (28 * 4) + ((3 - (storedTriangleId // 16)) * 4)
    ctrlDWord = s.dgfBuffer.Load(dword_offset)
    # Compute the bit index within the dword: each code uses 2 bits,
    # the codes are stored in reverse order (15 - (storedTriangleId % 16)).
    bitIdx = 2 * (15 - (storedTriangleId & 15))
    return (ctrlDWord >> bitIdx) & 3

def DGFGetVertex(s, vertexIndex):
    """
    Decode the vertex at index 'vertexIndex' from block s.
    
    The vertex bits (up to 48 bits) may span 3 dwords.
    We load 3 consecutive dwords starting at the dword containing the vertex,
    then align the bits using the helper 'align()' to obtain two aligned 32-bit words.
    The vertex components are then extracted:
      - x: lower 'bitsPerComponent.x' bits from the first dword,
      - y: next 'bitsPerComponent.y' bits (shifted out from the first dword),
      - z: remaining bits from the combined 64-bit value.
    Finally, we apply a mask based on the bits per component.
    """
    bitPos = s.vertexBitStart + vertexIndex * s.bitsPerVertex
    dwordPos = bitPos // 32
    # Load 3 consecutive 32-bit words starting at the appropriate byte offset.
    f = s.dgfBuffer.Load3(s.blockStartOffset + 4 * dwordPos)
    dw0 = align(f[0], f[1], bitPos)
    dw1 = align(f[1], f[2], bitPos)
    
    # Combine into a 64-bit value: lower 32 bits from dw0, upper 32 bits from dw1.
    vert = (dw1 << 32) | dw0
    
    bpc_x, bpc_y, bpc_z = s.header.bitsPerComponent
    # Extract components:
    v_x = dw0                   # lower bits: x coordinate
    v_y = dw0 >> bpc_x          # next bits: y coordinate
    v_z = vert >> (bpc_x + bpc_y) # remaining bits: z coordinate
    
    # Apply masks to keep only the valid bits.
    mask_x = (1 << bpc_x) - 1
    mask_y = (1 << bpc_y) - 1
    mask_z = (1 << bpc_z) - 1
    
    v_x = v_x & mask_x
    v_y = v_y & mask_y
    v_z = v_z & mask_z
    return (v_x, v_y, v_z)

def plot_histogram(data, title, xlabel, ylabel):
    """Plot a simple histogram of the given data using matplotlib."""
    plt.figure()
    plt.hist(data, bins=256, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def plot_histograms(header_dict):
    """Plot a histogram for each header property collected over all blocks."""
    keys = list(header_dict.keys())
    num_props = len(keys)
    # We'll create a grid with 5 columns:
    cols = 5
    rows = math.ceil(num_props / cols)
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 4 * rows))

    for i, key in enumerate(keys):
        ax = axes.flat[i]
        ax.hist(header_dict[key], bins=20, edgecolor='black')
        ax.set_title(key)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    # Remove any extra axes if present.
    for j in range(i + 1, rows * cols):
        fig.delaxes(axes.flat[j])
    plt.tight_layout()
    plt.show()

def plot_vertex_histograms(vertices_x, vertices_y, vertices_z):
    """Plot histograms for x, y, and z vertex positions in a single figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].hist(vertices_x, bins=20, edgecolor='black')
    axes[0].set_title('Vertex X')
    axes[0].set_xlabel('X Value')
    axes[0].set_ylabel('Frequency')
    
    axes[1].hist(vertices_y, bins=20, edgecolor='black')
    axes[1].set_title('Vertex Y')
    axes[1].set_xlabel('Y Value')
    axes[1].set_ylabel('Frequency')
    
    axes[2].hist(vertices_z, bins=20, edgecolor='black')
    axes[2].set_title('Vertex Z')
    axes[2].set_xlabel('Z Value')
    axes[2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

def main():
    csv_filename = "H:\Development\DGF-SDK\cmake-build-debug\data.csv"

    # Load the CSV data into a flat list of 32-bit uints.
    data = load_csv_as_uints(csv_filename)

    # Wrap the data into a ByteAddressBuffer.
    dgfBuffer = ByteAddressBuffer(data)

    # Each block is 128 bytes = 32 uints.
    num_blocks = len(data) // 32
    print(f"Found {num_blocks} block(s) in the CSV data.")

    # Create dictionaries to collect header property values.
    header_props = {
        'numTriangles': [],
        'numVerts': [],
        'bitsPerIndex': [],
        'bitsPerComponent.x': [],
        'bitsPerComponent.y': [],
        'bitsPerComponent.z': [],
        'anchor.x': [],
        'anchor.y': [],
        'anchor.z': [],
        'scale': [],
        'primIDBase': [],
        'userData': [],
        'bitSize': [],
        'geomIDMeta': [],
        'haveGeomIDPalette': []  # Will store as int (0 or 1)
    }

    # Iterate over each block, load header, and collect its properties.
    for block_index in range(num_blocks):
        blockInfo = DGFLoadBlockInfo(dgfBuffer, block_index)
        header = blockInfo.header
        header_props['numTriangles'].append(header.numTriangles)
        header_props['numVerts'].append(header.numVerts)
        header_props['bitsPerIndex'].append(header.bitsPerIndex)
        header_props['bitsPerComponent.x'].append(header.bitsPerComponent[0])
        header_props['bitsPerComponent.y'].append(header.bitsPerComponent[1])
        header_props['bitsPerComponent.z'].append(header.bitsPerComponent[2])
        header_props['anchor.x'].append(header.anchor[0])
        header_props['anchor.y'].append(header.anchor[1])
        header_props['anchor.z'].append(header.anchor[2])
        header_props['scale'].append(header.scale)
        header_props['primIDBase'].append(header.primIDBase)
        header_props['userData'].append(header.userData)
        header_props['bitSize'].append(header.bitSize)
        header_props['geomIDMeta'].append(header.geomIDMeta)
        header_props['haveGeomIDPalette'].append(1 if header.haveGeomIDPalette else 0)

    # Plot histograms for each property.
    #plot_histograms(header_props)

    # Collect triangle control values for all blocks.
    control_values = []
    for block_index in range(num_blocks):
        blockInfo = DGFLoadBlockInfo(dgfBuffer, block_index)
        numTriangles = blockInfo.header.numTriangles
        for t in range(numTriangles):
            ctrl = LoadTriangleControlValues(blockInfo, t)
            control_values.append(ctrl)

    # Pack 4 control values into a single uint.
    control_values_packed = []
    for i in range(0, len(control_values), 4):
        packed = 0
        for j in range(4):
            packed |= control_values[i + j] << (2 * j)
        control_values_packed.append(packed)
    
    # Plot the histogram of triangle control values.
    plot_histogram(control_values_packed, "Triangle Control Values", "Control Value", "Frequency")

    # --- Decode Vertex Positions ---
    vertices_x = []
    vertices_y = []
    vertices_z = []
    for block_index in range(num_blocks):
        blockInfo = DGFLoadBlockInfo(dgfBuffer, block_index)
        numVerts = blockInfo.header.numVerts
        for v in range(numVerts):
            vx, vy, vz = DGFGetVertex(blockInfo, v)
            vertices_x.append(vx)
            vertices_y.append(vy)
            vertices_z.append(vz)
    
    # Plot histograms for vertex positions.
    plot_vertex_histograms(vertices_x, vertices_y, vertices_z)

if __name__ == '__main__':
    main()

