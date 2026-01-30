#!/bin/bash
# Setup dataset script with optional vector normalization
# Usage: ./setup_dataset.sh <dataset_name> [normalize]
#   dataset_name: Name of the dataset directory (e.g., "glove", "nytimes")
#   normalize: Optional boolean parameter to normalize vectors (true/1 or false/0, default: false)

data=$1
normalize=${2:-false}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Dataset: $data"
echo "Normalize vectors: $normalize"

# Automatically detect dimension and number of vectors from base fvecs file
echo "Detecting dimension and number of vectors from ${data}/${data}_base.fvecs..."
fvecs_info=$(python3 $SCRIPT_DIR/get_fvecs_info.py $data/${data}_base.fvecs)
dim=$(echo $fvecs_info | cut -d' ' -f1)
num_rows=$(echo $fvecs_info | cut -d' ' -f2)

echo "Detected: dimension=$dim, num_vectors=$num_rows"

# Normalize vectors if requested
if [ "$normalize" = "true" ] || [ "$normalize" = "1" ]; then
    echo "Normalizing base and query fvecs files..."
    
    # Normalize base file
    if [ -f "$data/${data}_base.fvecs" ]; then
        echo "Normalizing base file: ${data}/${data}_base.fvecs"
        python3 $SCRIPT_DIR/normalize_fvecs.py $data/${data}_base.fvecs $data/${data}_base_normalized.fvecs
        if [ $? -ne 0 ]; then
            echo "Error: Failed to normalize base file"
            exit 1
        fi
        # Replace original with normalized version
        mv $data/${data}_base_normalized.fvecs $data/${data}_base.fvecs
        echo "Base file normalized successfully"
    else
        echo "Warning: Base file not found for normalization"
    fi
    
    # Normalize query file
    if [ -f "$data/${data}_query.fvecs" ]; then
        echo "Normalizing query file: ${data}/${data}_query.fvecs"
        python3 $SCRIPT_DIR/normalize_fvecs.py $data/${data}_query.fvecs $data/${data}_query_normalized.fvecs
        if [ $? -ne 0 ]; then
            echo "Error: Failed to normalize query file"
            exit 1
        fi
        # Replace original with normalized version
        mv $data/${data}_query_normalized.fvecs $data/${data}_query.fvecs
        echo "Query file normalized successfully"
    else
        echo "Warning: Query file not found for normalization"
    fi
    
    echo "Vector normalization completed"
else
    echo "Skipping vector normalization"
fi

# Check if query file exists, if not, skip concatenation
if [ -f "$data/${data}_query.fvecs" ]; then
    echo "Concatenating base and query fvecs files..."
    # For merged index
    python3 concatenate_fvecs.py $data/${data}_base.fvecs $data/${data}_query.fvecs $data/${data}_all.fvecs
    # For merged index + switching X and Y
    python3 concatenate_fvecs.py $data/${data}_query.fvecs $data/${data}_base.fvecs $data/${data}_reverse_all.fvecs
else
    echo "Query file not found, skipping concatenation"
    exit 1
fi

# Build indices for each type
for type in base query all; do
    fvec_file="$data/${data}_${type}.fvecs"
    
    if [ -f "$fvec_file" ]; then
        echo "Building index for ${type}..."
        
        # 1. Get the number of vectors for this specific file
        current_info=$(python3 $SCRIPT_DIR/get_fvecs_info.py "$data/${data}_base.fvecs")
        seed_offset=$(echo $current_info | cut -d' ' -f2)
        
        # 2. Set OOD parameters only for the 'all' type
        if [ "$type" = "all" ]; then
            ood_file="${data}/${data}_ood.txt"
            enable_ood=1
        else
            # For base/query, we use placeholders or dummy values since OOD is disabled
            ood_file="${data}/unused.txt"
            enable_ood=0
        fi

        # 3. Call the build script with extended arguments
        # Parameters passed: input_fvecs, temp_graph, output_nsg, num_rows, ood_txt_path, ood_flag
        bash build_nsg.sh "$fvec_file" "temp.graph" "$data/${data}_${type}.nsg" "$seed_offset" "$ood_file" 1 #"$enable_ood"
    else
        echo "Skipping ${type} - file not found"
        exit 1
    fi
done

# Run clustering
#echo "Running clustering..."
#python3 simple_cluster_binary.py $data/${data}_base.fvecs $num_rows $dim 1000 $data/clusters_1000
#python3 convert_clusters_pkl_to_txt.py $data/clusters_1000/clusters.pkl $data/clusters_1000/clusters.txt

echo "Setup completed successfully!"
