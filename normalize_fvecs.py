#!/usr/bin/env python3
"""
Standalone script to normalize fvecs files for cosine similarity.
Extracted from build_nsg_cosine_robust.sh

Usage:
    python3 normalize_fvecs.py input.fvecs [output.fvecs]
    
If output file is not specified, it will be named input_normalized.fvecs
"""

import numpy as np
import struct
import sys
import os

def normalize_fvecs(input_file, output_file):
    """
    Normalize vectors in an fvecs file to unit length.
    
    Args:
        input_file (str): Path to input fvecs file
        output_file (str): Path to output normalized fvecs file
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return False
    
    print(f"Reading vectors from {input_file}...")
    
    with open(input_file, 'rb') as f:
        # Read all vectors
        vectors = []
        dim = None
        while True:
            # Read dimension for each vector
            dim_bytes = f.read(4)
            if len(dim_bytes) == 0:
                break
            if len(dim_bytes) != 4:
                break
                
            dim = struct.unpack('<I', dim_bytes)[0]
                
            # Read vector data
            vector_bytes = f.read(dim * 4)
            if len(vector_bytes) != dim * 4:
                break
                
            vector = struct.unpack('<' + 'f' * dim, vector_bytes)
            vectors.append(np.array(vector, dtype=np.float32))
    
    if not vectors:
        print("Error: No vectors found in input file.")
        return False
    
    print(f'Loaded {len(vectors)} vectors of dimension {dim}')
    
    # Check if we have enough vectors
    if len(vectors) < 10:
        print(f'Warning: Only {len(vectors)} vectors found. This might be a very small dataset.')
        print('Consider using a larger dataset for better results.')
    
    # Normalize vectors
    print("Normalizing vectors...")
    normalized_vectors = []
    for i, vec in enumerate(vectors):
        norm = np.linalg.norm(vec)
        if norm > 0:
            normalized_vec = vec / norm
        else:
            normalized_vec = vec  # Keep zero vectors as is
        normalized_vectors.append(normalized_vec)
        
        if (i + 1) % 10000 == 0:
            print(f'  Normalized {i + 1} vectors...')
    
    # Write normalized vectors
    print(f"Writing normalized vectors to {output_file}...")
    with open(output_file, 'wb') as f:
        for vec in normalized_vectors:
            f.write(struct.pack('<I', dim))
            f.write(struct.pack('<' + 'f' * dim, *vec))
    
    print(f'Successfully written {len(normalized_vectors)} normalized vectors to {output_file}')
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 normalize_fvecs.py input.fvecs [output.fvecs]")
        print("If output file is not specified, it will be named input_normalized.fvecs")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Generate output filename by adding _normalized before .fvecs
        if input_file.endswith('.fvecs'):
            output_file = input_file[:-6] + '_normalized.fvecs'
        else:
            output_file = input_file + '_normalized.fvecs'
    
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()
    
    success = normalize_fvecs(input_file, output_file)
    if not success:
        sys.exit(1)
    
    print("\nNormalization completed successfully!")
    print(f"Normalized vectors saved to: {output_file}")

if __name__ == '__main__':
    main()
