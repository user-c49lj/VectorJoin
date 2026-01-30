#!/usr/bin/env python3
"""
Get dimension and number of vectors from a fvecs file
"""

import struct
import sys
from pathlib import Path

def get_fvecs_info(filepath):
    """Get dimension and number of vectors from fvecs file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File {filepath} does not exist")
    
    num_vectors = 0
    dimension = None
    
    with open(filepath, 'rb') as f:
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            
            dim = struct.unpack('<I', dim_bytes)[0]
            
            # Set dimension from first vector
            if dimension is None:
                dimension = dim
            
            # Verify dimension consistency
            if dim != dimension:
                raise ValueError(f"Inconsistent dimensions: expected {dimension}, got {dim}")
            
            # Read vector data
            vector_bytes = f.read(dim * 4)
            if len(vector_bytes) != dim * 4:
                raise ValueError(f"Incomplete vector data at vector {num_vectors}")
            
            num_vectors += 1
    
    return dimension, num_vectors

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 get_fvecs_info.py <fvecs_file>")
        print("Output: dimension num_vectors")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    try:
        dimension, num_vectors = get_fvecs_info(filepath)
        print(f"{dimension} {num_vectors}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()