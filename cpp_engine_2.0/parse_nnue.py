import struct
import sys

def parse_nnue(path):
    with open(path, 'rb') as f:
        # NNUE files start with a magic version hash (4 bytes)
        version = struct.unpack('<I', f.read(4))[0]
        print(f"Version Hash: {hex(version)}")
        
        # Then usually a description string
        desc_len = struct.unpack('<I', f.read(4))[0]
        desc = f.read(desc_len).decode('utf-8', errors='ignore')
        print(f"Description: {desc}")
        
        # Then the feature transformer (HalfKAv2)
        # Magic hash for feature transformer
        ft_hash = struct.unpack('<I', f.read(4))[0]
        print(f"FT Hash: {hex(ft_hash)}")
        
        # Read biases and weights for HalfKAv2
        # Dimensions are usually 256 for biases
        # and 45056 * 256 for weights (or 45056 * 256 * 2 for HalfKAv2)

if __name__ == '__main__':
    # Trying to parse a sample if we had one
    # parse_nnue('brain.nnue')
    pass
