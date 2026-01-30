#!/usr/bin/env python3
"""
Program to concatenate two fvecs files into one.

The fvecs format stores vectors as:
- 4 bytes: dimension (int32)
- dimension * 4 bytes: vector data (float32)

This program reads two fvecs files and concatenates them into a single output file.
"""

import struct
import argparse
import sys
import os

def read_fvecs_info(filename):
    """Read basic information about an fvecs file (dimension, count)"""
    with open(filename, 'rb') as f:
        # Read first vector to get dimension
        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            return 0, 0
        
        dim = struct.unpack('i', dim_bytes)[0]
        
        # Count total vectors
        count = 0
        f.seek(0)  # Reset to beginning
        
        while True:
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                break
            current_dim = struct.unpack('i', dim_bytes)[0]
            
            if current_dim != dim:
                raise ValueError(f"Inconsistent dimension in {filename}: expected {dim}, got {current_dim}")
            
            # Skip vector data
            f.seek(dim * 4, 1)
            count += 1
        
        return dim, count

def concatenate_fvecs(file1, file2, output_file):
    """Concatenate two fvecs files into one output file"""
    
    # Validate input files exist
    if not os.path.exists(file1):
        raise FileNotFoundError(f"Input file 1 not found: {file1}")
    if not os.path.exists(file2):
        raise FileNotFoundError(f"Input file 2 not found: {file2}")
    
    # Get information about both files
    print(f"Analyzing {file1}...")
    dim1, count1 = read_fvecs_info(file1)
    print(f"  Dimension: {dim1}, Vectors: {count1}")
    
    print(f"Analyzing {file2}...")
    dim2, count2 = read_fvecs_info(file2)
    print(f"  Dimension: {dim2}, Vectors: {count2}")
    
    # Validate dimensions match
    if dim1 != dim2:
        raise ValueError(f"Dimension mismatch: {file1} has dimension {dim1}, {file2} has dimension {dim2}")
    
    print(f"Concatenating {count1} + {count2} = {count1 + count2} vectors...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Copy all vectors from both files
    with open(output_file, 'wb') as outf:
        # Copy vectors from first file
        with open(file1, 'rb') as f1:
            while True:
                chunk = f1.read(4096)  # Read in chunks for efficiency
                if not chunk:
                    break
                outf.write(chunk)
        
        # Copy vectors from second file
        with open(file2, 'rb') as f2:
            while True:
                chunk = f2.read(4096)  # Read in chunks for efficiency
                if not chunk:
                    break
                outf.write(chunk)
    
    print(f"Successfully created {output_file}")
    print(f"Total vectors: {count1 + count2}, Dimension: {dim1}")

def main():
    parser = argparse.ArgumentParser(
        description="Concatenate two fvecs files into one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python concatenate_fvecs.py file1.fvecs file2.fvecs output.fvecs
  python concatenate_fvecs.py sift/sift_base.fvecs sift/sift_query.fvecs combined.fvecs
        """
    )
    
    parser.add_argument('file1', help='First input fvecs file')
    parser.add_argument('file2', help='Second input fvecs file')
    parser.add_argument('output', help='Output fvecs file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        concatenate_fvecs(args.file1, args.file2, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()