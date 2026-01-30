#!/usr/bin/env python3
"""
ANN-Benchmarks & VIBE Dataset Downloader and Converter
Supports standard HDF5 datasets and VIBE (Hugging Face) datasets.
Converts to .fvecs and .ivecs formats.
"""

import os
import h5py
import numpy as np
import struct
from pathlib import Path
import argparse
import logging
import json
from typing import Dict
import subprocess

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnnBenchmarksDownloader:
    def __init__(self, data_dir="ann_benchmarks_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Complete Dataset Registry
        self.datasets = {
            # --- Standard ANN-Benchmarks ---
            'fashion-mnist-784-euclidean': {'url': 'http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5', 'description': 'Fashion-MNIST (784 dim, Euclidean)', 'size': '217MB', 'suitable_for_fvecs': True},
            'mnist-784-euclidean': {'url': 'http://ann-benchmarks.com/mnist-784-euclidean.hdf5', 'description': 'MNIST (784 dim, Euclidean)', 'size': '217MB', 'suitable_for_fvecs': True},
            'sift-128-euclidean': {'url': 'http://ann-benchmarks.com/sift-128-euclidean.hdf5', 'description': 'SIFT (128 dim, Euclidean)', 'size': '501MB', 'suitable_for_fvecs': True},
            'gist-960-euclidean': {'url': 'http://ann-benchmarks.com/gist-960-euclidean.hdf5', 'description': 'GIST (960 dim, Euclidean)', 'size': '3.6GB', 'suitable_for_fvecs': True},
            'glove-25-angular': {'url': 'http://ann-benchmarks.com/glove-25-angular.hdf5', 'description': 'GloVe (25 dim, Angular)', 'size': '121MB', 'suitable_for_fvecs': True},
            'glove-50-angular': {'url': 'http://ann-benchmarks.com/glove-50-angular.hdf5', 'description': 'GloVe (50 dim, Angular)', 'size': '235MB', 'suitable_for_fvecs': True},
            'glove-100-angular': {'url': 'http://ann-benchmarks.com/glove-100-angular.hdf5', 'description': 'GloVe (100 dim, Angular)', 'size': '463MB', 'suitable_for_fvecs': True},
            'glove-200-angular': {'url': 'http://ann-benchmarks.com/glove-200-angular.hdf5', 'description': 'GloVe (200 dim, Angular)', 'size': '918MB', 'suitable_for_fvecs': True},
            'nytimes-256-angular': {'url': 'http://ann-benchmarks.com/nytimes-256-angular.hdf5', 'description': 'NYTimes (256 dim, Angular)', 'size': '301MB', 'suitable_for_fvecs': True},
            'lastfm-64-dot': {'url': 'http://ann-benchmarks.com/lastfm-64-dot.hdf5', 'description': 'Last.fm (64 dim, Dot product)', 'size': '135MB', 'suitable_for_fvecs': True},
            'kosarak-jaccard': {'url': 'http://ann-benchmarks.com/kosarak-jaccard.hdf5', 'description': 'Kosarak (Jaccard) - SPARSE', 'size': '33MB', 'suitable_for_fvecs': False},
            'movielens10m-jaccard': {'url': 'http://ann-benchmarks.com/movielens10m-jaccard.hdf5', 'description': 'MovieLens (Jaccard) - SPARSE', 'size': '63MB', 'suitable_for_fvecs': False},
            
            # --- VIBE Benchmark Datasets ---
            'coco-nomic-768-normalized': {'url': 'https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/coco-nomic-768-normalized.hdf5', 'description': 'VIBE: COCO Nomic (768 dim, Normalized)', 'size': '~850MB', 'suitable_for_fvecs': True},
            'imagenet-align-640-normalized': {'url': 'https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/imagenet-align-640-normalized.hdf5', 'description': 'VIBE: ImageNet Align (640 dim, Normalized)', 'size': '~3.1GB', 'suitable_for_fvecs': True},
            'laion-clip-512-normalized': {'url': 'https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/laion-clip-512-normalized.hdf5', 'description': 'VIBE: LAION CLIP (512 dim, Normalized)', 'size': '~2GB', 'suitable_for_fvecs': True}
        }

    def list_datasets(self):
        print("\nAvailable datasets (ANN-Benchmarks & VIBE):")
        print("=" * 115)
        print(f"{'Dataset Name':<35} | {'Description':<55} | {'Size'}")
        print("=" * 115)
        for name, info in self.datasets.items():
            print(f"{name:<35} | {info['description']:<55} | {info['size']}")
        print("=" * 115)

    def download_file(self, url: str, filename: str) -> Path:
        filepath = self.data_dir / filename
        if filepath.exists():
            logger.info(f"File {filename} already exists in {self.data_dir}, skipping download")
            return filepath
            
        logger.info(f"Downloading {filename} to {self.data_dir}...")
        try:
            cmd = ["wget", "--user-agent=Mozilla/5.0", "--show-progress", "-O", str(filepath), url]
            subprocess.run(cmd, check=True)
            return filepath
        except Exception as e:
            logger.error(f"Wget failed: {e}")
            if filepath.exists(): filepath.unlink()
            raise

    def read_hdf5_dataset(self, filepath: Path) -> Dict:
        logger.info(f"Reading HDF5: {filepath}")
        data = {}
        with h5py.File(filepath, 'r') as f:
            # Check for standard ANN-Benchmarks keys vs VIBE keys
            if 'train' in f: data['train'] = np.array(f['train'])
            elif 'corpus' in f: data['train'] = np.array(f['corpus'])

            if 'test' in f: data['test'] = np.array(f['test'])
            elif 'queries' in f: data['test'] = np.array(f['queries'])

            if 'neighbors' in f: data['neighbors'] = np.array(f['neighbors'])
            if 'distances' in f: data['distances'] = np.array(f['distances'])
            
            data['metadata'] = {k: v for k, v in f.attrs.items()}
            return data

    def write_fvecs(self, vectors: np.ndarray, filename: str) -> Path:
        filepath = self.data_dir / filename
        logger.info(f"Writing {vectors.shape[0]} vectors to {filename}")
        with open(filepath, 'wb') as f:
            for vector in vectors:
                f.write(struct.pack('<I', vector.shape[0]))
                f.write(vector.astype(np.float32).tobytes())
        return filepath

    def write_ivecs(self, vectors: np.ndarray, filename: str) -> Path:
        filepath = self.data_dir / filename
        logger.info(f"Writing {vectors.shape[0]} vectors to {filename}")
        with open(filepath, 'wb') as f:
            for vector in vectors:
                f.write(struct.pack('<I', vector.shape[0]))
                f.write(vector.astype(np.uint32).tobytes())
        return filepath

    def process_dataset(self, dataset_name: str):
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        info = self.datasets[dataset_name]
        hdf5_filename = f"{dataset_name}.hdf5"
        hdf5_path = self.download_file(info['url'], hdf5_filename)
        
        data = self.read_hdf5_dataset(hdf5_path)
        results = {}

        # Only convert if the dataset is dense/suitable
        if info['suitable_for_fvecs']:
            if 'train' in data:
                results['train_fvecs'] = self.write_fvecs(data['train'], f"{dataset_name}_train.fvecs")
            if 'test' in data:
                results['test_fvecs'] = self.write_fvecs(data['test'], f"{dataset_name}_test.fvecs")
        else:
            logger.warning(f"Dataset {dataset_name} is marked as sparse. Skipping fvecs conversion.")

        if 'neighbors' in data:
            results['neighbors_ivecs'] = self.write_ivecs(data['neighbors'], f"{dataset_name}_neighbors.ivecs")

        return results

def main():
    parser = argparse.ArgumentParser(description='Download ANN datasets and convert to fvecs')
    parser.add_argument('dataset', nargs='?', help='Dataset name to download')
    parser.add_argument('--data-dir', default='ann_benchmarks_data', help='Directory to store data (default: ann_benchmarks_data)')
    parser.add_argument('--list', action='store_true', help='List all available datasets')
    args = parser.parse_args()

    # Initialize with the user-provided data directory
    downloader = AnnBenchmarksDownloader(data_dir=args.data_dir)

    if args.list:
        downloader.list_datasets()
        return

    if not args.dataset:
        print("Error: No dataset specified. Use --list to see available datasets.")
        return

    try:
        downloader.process_dataset(args.dataset)
        print(f"\nSuccessfully processed: {args.dataset}")
        print(f"Files are located in: {Path(args.data_dir).resolve()}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()