#ifndef SIMJOIN_COMMON_H
#define SIMJOIN_COMMON_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <deque>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cmath>
#include <random>
#include <set>
#include <iomanip>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <limits>
#include <filesystem>
#include <cassert>
#include <sstream>
#include <numeric>

// Hash function for std::pair<int, int>
namespace std {
    template<>
    struct hash<std::pair<int, int>> {
        size_t operator()(const std::pair<int, int>& p) const {
            return hash<int>()(p.first) ^ (hash<int>()(p.second) << 1);
        }
    };
}

// ----------------------------------------------------------------------
// Basic utilities
// ----------------------------------------------------------------------

/**
 * Euclidean distance (‖a-b‖₂) - L2 norm
 * Used throughout the algorithm for distance computations
 */
double l2_distance(const std::vector<float>& a, const std::vector<float>& b) {
    // Increment thread-local distance computation counter if it exists
    extern thread_local long long thread_distance_computations;
    thread_distance_computations++;
    
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector dimensions don't match");
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

/**
 * Cosine distance (1 - cosine_similarity)
 * Used for cosine similarity-based distance computations
 * Assumes vectors are already normalized (unit vectors)
 */
double cosine_distance(const std::vector<float>& a, const std::vector<float>& b) {
    // Increment thread-local distance computation counter if it exists
    extern thread_local long long thread_distance_computations;
    thread_distance_computations++;
    
    if (a.size() != b.size()) {
        throw std::runtime_error("Vector dimensions don't match");
    }
    
    double dot_product = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
    }
    
    // For normalized vectors, cosine similarity = dot product
    // Cosine distance = 1 - cosine similarity
    return 1.0 - dot_product;
}

/**
 * Graph class for representing proximity graphs
 * Used for both X and Y graphs in the SimJoin algorithm
 */
class Graph {
private:
    std::unordered_map<int, std::vector<std::pair<int, double>>> neighbors;

public:
    void add_edge(int u, int v, double weight) {
        neighbors[u].emplace_back(v, weight);
    }
    
    // Online edge addition with duplicate checking (for query-to-data edges)
    void add_edge_online(int u, int v, double weight) {
        // Check if edge already exists to avoid duplicates
        auto& u_neighbors = neighbors[u];
        for (const auto& [existing_v, existing_weight] : u_neighbors) {
            if (existing_v == v) {
                // Edge already exists, skip adding duplicate edge
                return;
            }
        }
        // Edge doesn't exist, add it
        u_neighbors.emplace_back(v, weight);
    }
    
    const std::vector<std::pair<int, double>>& get_neighbors(int u) const {
        try {
            return neighbors.at(u);
        } catch (const std::out_of_range&) {
            static const std::vector<std::pair<int, double>> empty;
            return empty;
        }
    }
    
    std::vector<int> get_nodes() const {
        std::vector<int> nodes;
        for (const auto& [node, _] : neighbors) {
            nodes.push_back(node);
        }
        return nodes;
    }
    
    size_t size() const {
        return neighbors.size();
    }
    
    bool has_node(int u) const {
        return neighbors.count(u) > 0;
    }
};

// ----------------------------------------------------------------------
// Data loading utilities
// ----------------------------------------------------------------------

/**
 * Load vectors from fvecs file format
 */
std::unordered_map<int, std::vector<float>> load_fvecs_vectors(const std::string& filename, int max_vectors = -1) {
    std::cout << "Loading vectors from fvecs file: " << filename << "..." << std::endl;
    
    std::unordered_map<int, std::vector<float>> vectors;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open fvecs file: " + filename);
    }
    
    int vector_id = 0;
    while (max_vectors == -1 || vector_id < max_vectors) {
        // Read dimension from fvecs format (4-byte integer)
        int32_t vec_dim;
        file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
        
        if (file.eof()) {
            std::cout << "Reached end of file after " << vector_id << " vectors" << std::endl;
            break;
        }
        
        // Read vector data
        std::vector<float> vector_data(vec_dim);
        file.read(reinterpret_cast<char*>(vector_data.data()), vec_dim * sizeof(float));
        
        if (static_cast<size_t>(file.gcount()) == vec_dim * sizeof(float)) {
            vectors[vector_id] = vector_data;
        } else {
            std::cout << "Reached end of file while reading vector " << vector_id << std::endl;
            break;
        }
        
        vector_id++;
    }
    
    std::cout << "Loaded " << vectors.size() << " vectors from fvecs file" << std::endl;
    return vectors;
}

/**
 * Load CAGRA index from binary format using the correct CAGRA deserialization
 * Based on simjoin_parallel_profiled.cpp implementation
 */
Graph load_cagra_index(const std::string& filename) {
    std::cout << "Loading CAGRA index from: " << filename << "..." << std::endl;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CAGRA index file: " + filename);
    }
    
    // Read dtype string (4 bytes)
    char dtype_str[4];
    file.read(dtype_str, 4);
    if (file.gcount() != 4) {
        throw std::runtime_error("Cannot read dtype string");
    }
    std::cout << "  Dtype: " << std::string(dtype_str, 4) << std::endl;
    
    // Helper function to read NumPy scalar
    auto read_numpy_scalar = [&file](const std::string& expected_dtype) -> uint64_t {
        // Read magic string
        char magic[6];
        file.read(magic, 6);
        if (std::string(magic, 6) != "\x93NUMPY") {
            throw std::runtime_error("Invalid NumPy magic string");
        }
        
        // Read version
        uint8_t major_version = file.get();
        uint8_t minor_version = file.get();
        
        // Read header length
        uint16_t header_len;
        file.read(reinterpret_cast<char*>(&header_len), 2);
        
        // Read header
        std::string header(header_len, '\0');
        file.read(&header[0], header_len);
        
        // Parse header to get dtype
        size_t descr_pos = header.find("'descr': '");
        if (descr_pos == std::string::npos) {
            throw std::runtime_error("Cannot find descr in header");
        }
        descr_pos += 10;
        size_t descr_end = header.find("'", descr_pos);
        std::string descr = header.substr(descr_pos, descr_end - descr_pos);
        
        if (descr != expected_dtype) {
            std::cout << "Warning: Expected dtype " << expected_dtype << ", got " << descr << std::endl;
        }
        
        // Read the actual data based on dtype
        if (descr == "<i4") {
            int32_t value;
            file.read(reinterpret_cast<char*>(&value), 4);
            return value;
        } else if (descr == "<u4") {
            uint32_t value;
            file.read(reinterpret_cast<char*>(&value), 4);
            return value;
        } else if (descr == "<u8") {
            uint64_t value;
            file.read(reinterpret_cast<char*>(&value), 8);
            return value;
        } else if (descr == "?") {
            bool value;
            file.read(reinterpret_cast<char*>(&value), 1);
            return value ? 1 : 0;
        } else if (descr == "|u1") {
            uint8_t value;
            file.read(reinterpret_cast<char*>(&value), 1);
            return value;
        } else {
            throw std::runtime_error("Unsupported dtype: " + descr);
        }
    };
    
    // Helper function to read NumPy array
    auto read_numpy_array = [&file]() -> std::vector<uint32_t> {
        // Read magic string
        char magic[6];
        file.read(magic, 6);
        if (std::string(magic, 6) != "\x93NUMPY") {
            throw std::runtime_error("Invalid NumPy magic string");
        }
        
        // Read version
        uint8_t major_version = file.get();
        uint8_t minor_version = file.get();
        
        // Read header length
        uint16_t header_len;
        file.read(reinterpret_cast<char*>(&header_len), 2);
        
        // Read header
        std::string header(header_len, '\0');
        file.read(&header[0], header_len);
        
        // Parse header to get shape
        size_t shape_pos = header.find("'shape': (");
        if (shape_pos == std::string::npos) {
            throw std::runtime_error("Cannot find shape in header");
        }
        shape_pos += 10;
        size_t shape_end = header.find(")", shape_pos);
        std::string shape_str = header.substr(shape_pos, shape_end - shape_pos);
        
        // Parse shape
        std::vector<uint64_t> shape;
        std::stringstream ss(shape_str);
        std::string dim;
        while (std::getline(ss, dim, ',')) {
            shape.push_back(std::stoul(dim));
        }
        
        // Calculate total elements
        uint64_t total_elements = 1;
        for (uint64_t dim : shape) {
            total_elements *= dim;
        }
        
        // Read array data
        std::vector<uint32_t> data;
        data.reserve(total_elements);
        for (uint64_t i = 0; i < total_elements; ++i) {
            uint32_t value;
            file.read(reinterpret_cast<char*>(&value), 4);
            data.push_back(value);
        }
        
        return data;
    };
    
    // Read metadata
    uint64_t version = read_numpy_scalar("<i4");
    uint64_t dataset_size = read_numpy_scalar("<u4");
    uint64_t dimension = read_numpy_scalar("<u4");
    uint64_t graph_degree = read_numpy_scalar("<u4");
    uint64_t metric_type = read_numpy_scalar("<u4");
    
    std::cout << "  Version: " << version << std::endl;
    std::cout << "  Dataset size: " << dataset_size << std::endl;
    std::cout << "  Dimension: " << dimension << std::endl;
    std::cout << "  Graph degree: " << graph_degree << std::endl;
    std::cout << "  Metric type: " << metric_type << std::endl;
    
    // Read graph data
    std::vector<uint32_t> graph_data = read_numpy_array();
    std::cout << "  Graph data elements: " << graph_data.size() << std::endl;
    
    // Read has_dataset flag
    uint64_t has_dataset = read_numpy_scalar("|u1");
    std::cout << "  Has dataset: " << has_dataset << std::endl;
    
    // Build graph from CAGRA data
    Graph graph;
    
    // Process graph edges
    // The graph_data is stored as a 2D array with shape (dataset_size, graph_degree)
    // Each row contains neighbors for one node, padded with UINT32_MAX if needed
    uint64_t graph_degree_actual = graph_data.size() / dataset_size;
    for (uint64_t i = 0; i < dataset_size; ++i) {
        for (uint64_t j = 0; j < graph_degree_actual; ++j) {
            uint32_t neighbor = graph_data[i * graph_degree_actual + j];
            if (neighbor != 0xFFFFFFFF) {  // Filter out sentinel values
                graph.add_edge(i, neighbor, 0.0);  // Placeholder weight
            }
        }
    }
    
    std::cout << "Loaded CAGRA index with " << graph.size() << " nodes" << std::endl;
    return graph;
}

/**
 * Load CAGRA format data directly from binary file (returns both vectors and neighbors)
 * This is the version used by simjoin_parallel_profiled.cpp
 */
std::pair<std::unordered_map<int, std::vector<float>>, std::unordered_map<int, std::vector<int>>> 
load_cagra_data_direct(const std::string& cagra_file) {
    std::cout << "Loading CAGRA file: " << cagra_file << std::endl;
    
    std::ifstream file(cagra_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CAGRA file: " + cagra_file);
    }
    
    // Read dtype string (4 bytes)
    char dtype_str[4];
    file.read(dtype_str, 4);
    if (file.gcount() != 4) {
        throw std::runtime_error("Cannot read dtype string");
    }
    std::cout << "  Dtype: " << std::string(dtype_str, 4) << std::endl;
    
    // Helper function to read NumPy scalar
    auto read_numpy_scalar = [&file](const std::string& expected_dtype) -> uint64_t {
        // Read magic string
        char magic[6];
        file.read(magic, 6);
        if (std::string(magic, 6) != "\x93NUMPY") {
            throw std::runtime_error("Invalid NumPy magic string");
        }
        
        // Read version
        uint8_t major_version = file.get();
        uint8_t minor_version = file.get();
        
        // Read header length
        uint16_t header_len;
        file.read(reinterpret_cast<char*>(&header_len), 2);
        
        // Read header
        std::string header(header_len, '\0');
        file.read(&header[0], header_len);
        
        // Parse header to get dtype
        size_t descr_pos = header.find("'descr': '");
        if (descr_pos == std::string::npos) {
            throw std::runtime_error("Cannot find descr in header");
        }
        descr_pos += 10;
        size_t descr_end = header.find("'", descr_pos);
        std::string descr = header.substr(descr_pos, descr_end - descr_pos);
        
        if (descr != expected_dtype) {
            std::cout << "Warning: Expected dtype " << expected_dtype << ", got " << descr << std::endl;
        }
        
        // Read the actual data based on dtype
        if (descr == "<i4") {
            int32_t value;
            file.read(reinterpret_cast<char*>(&value), 4);
            return value;
        } else if (descr == "<u4") {
            uint32_t value;
            file.read(reinterpret_cast<char*>(&value), 4);
            return value;
        } else if (descr == "<u8") {
            uint64_t value;
            file.read(reinterpret_cast<char*>(&value), 8);
            return value;
        } else if (descr == "?") {
            bool value;
            file.read(reinterpret_cast<char*>(&value), 1);
            return value ? 1 : 0;
        } else if (descr == "|u1") {
            uint8_t value;
            file.read(reinterpret_cast<char*>(&value), 1);
            return value;
        } else {
            throw std::runtime_error("Unsupported dtype: " + descr);
        }
    };
    
    // Helper function to read NumPy array
    auto read_numpy_array = [&file]() -> std::vector<uint32_t> {
        // Read magic string
        char magic[6];
        file.read(magic, 6);
        if (std::string(magic, 6) != "\x93NUMPY") {
            throw std::runtime_error("Invalid NumPy magic string");
        }
        
        // Read version
        uint8_t major_version = file.get();
        uint8_t minor_version = file.get();
        
        // Read header length
        uint16_t header_len;
        file.read(reinterpret_cast<char*>(&header_len), 2);
        
        // Read header
        std::string header(header_len, '\0');
        file.read(&header[0], header_len);
        
        // Parse header to get shape
        size_t shape_pos = header.find("'shape': (");
        if (shape_pos == std::string::npos) {
            throw std::runtime_error("Cannot find shape in header");
        }
        shape_pos += 10;
        size_t shape_end = header.find(")", shape_pos);
        std::string shape_str = header.substr(shape_pos, shape_end - shape_pos);
        
        // Parse shape
        std::vector<uint64_t> shape;
        std::stringstream ss(shape_str);
        std::string dim;
        while (std::getline(ss, dim, ',')) {
            shape.push_back(std::stoul(dim));
        }
        
        // Calculate total elements
        uint64_t total_elements = 1;
        for (uint64_t dim : shape) {
            total_elements *= dim;
        }
        
        // Read array data
        std::vector<uint32_t> data;
        data.reserve(total_elements);
        for (uint64_t i = 0; i < total_elements; ++i) {
            uint32_t value;
            file.read(reinterpret_cast<char*>(&value), 4);
            data.push_back(value);
        }
        
        return data;
    };
    
    // Read metadata
    uint64_t version = read_numpy_scalar("<i4");
    uint64_t dataset_size = read_numpy_scalar("<u4");
    uint64_t dimension = read_numpy_scalar("<u4");
    uint64_t graph_degree = read_numpy_scalar("<u4");
    uint64_t metric_type = read_numpy_scalar("<u4");
    
    std::cout << "  Version: " << version << std::endl;
    std::cout << "  Dataset size: " << dataset_size << std::endl;
    std::cout << "  Dimension: " << dimension << std::endl;
    std::cout << "  Graph degree: " << graph_degree << std::endl;
    std::cout << "  Metric type: " << metric_type << std::endl;
    
    // Read graph data
    std::vector<uint32_t> graph_data = read_numpy_array();
    std::cout << "  Graph data elements: " << graph_data.size() << std::endl;
    
    // Read has_dataset flag
    uint64_t has_dataset = read_numpy_scalar("|u1");
    std::cout << "  Has dataset: " << has_dataset << std::endl;
    
    // Process graph data
    std::unordered_map<int, std::vector<float>> vectors;
    std::unordered_map<int, std::vector<int>> graph_edges;
    
    // Initialize vectors (empty for now, will be filled by embedding files)
    for (uint64_t i = 0; i < dataset_size; ++i) {
        vectors[i] = std::vector<float>();
    }
    
    // Process graph edges
    // The graph_data is stored as a 2D array with shape (dataset_size, graph_degree)
    // Each row contains neighbors for one node, padded with UINT32_MAX if needed
    uint64_t graph_degree_actual = graph_data.size() / dataset_size;
    for (uint64_t i = 0; i < dataset_size; ++i) {
        std::vector<int> neighbors;
        for (uint64_t j = 0; j < graph_degree_actual; ++j) {
            uint32_t neighbor = graph_data[i * graph_degree_actual + j];
            if (neighbor != 0xFFFFFFFF) {  // Filter out sentinel values
                neighbors.push_back(neighbor);
            }
        }
        graph_edges[i] = neighbors;
    }
    
    std::cout << "Loaded " << vectors.size() << " vectors with neighbors" << std::endl;
    return {vectors, graph_edges};
}

/**
 * Load NSG format data directly from binary file (returns vectors, neighbors, and entry point)
 * This avoids the CAGRA padding issue by reading the compact NSG format directly
 */
std::tuple<std::unordered_map<int, std::vector<float>>, std::unordered_map<int, std::vector<int>>, int> 
load_nsg_data_direct(const std::string& nsg_file) {
    std::cout << "Loading NSG file: " << nsg_file << std::endl;
    
    std::ifstream file(nsg_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open NSG file: " + nsg_file);
    }
    
    // Read NSG header
    uint32_t width, ep_;
    file.read(reinterpret_cast<char*>(&width), 4);
    file.read(reinterpret_cast<char*>(&ep_), 4);
    
    std::cout << "  NSG width: " << width << std::endl;
    std::cout << "  NSG entry point: " << ep_ << std::endl;
    
    // Process graph data
    std::unordered_map<int, std::vector<float>> vectors;
    std::unordered_map<int, std::vector<int>> graph_edges;
    
    uint32_t vector_count = 0;
    while (true) {
        // Try to read k (degree)
        uint32_t k;
        file.read(reinterpret_cast<char*>(&k), 4);
        if (file.gcount() != 4) {
            break; // End of file
        }
        
        // Read k neighbor indices
        std::vector<int> neighbors;
        neighbors.reserve(k);
        
        for (uint32_t i = 0; i < k; ++i) {
            uint32_t neighbor;
            file.read(reinterpret_cast<char*>(&neighbor), 4);
            if (file.gcount() != 4) {
                break; // Incomplete data
            }
            neighbors.push_back(neighbor);
        }
        
        if (neighbors.size() == k) {
            // Initialize empty vector (will be filled by embedding files)
            vectors[vector_count] = std::vector<float>();
            graph_edges[vector_count] = neighbors;
            vector_count++;
            
            if (vector_count % 10000 == 0) {
                std::cout << "  Processed " << vector_count << " vectors..." << std::endl;
            }
        } else {
            break; // Incomplete data
        }
    }
    
    std::cout << "  Total vectors: " << vector_count << std::endl;
    std::cout << "  Average degree: " << (vector_count > 0 ? 
        (double)std::accumulate(graph_edges.begin(), graph_edges.end(), 0, 
            [](int sum, const auto& pair) { return sum + pair.second.size(); }) / vector_count : 0.0) << std::endl;
    
    std::cout << "Loaded " << vectors.size() << " vectors with neighbors from NSG" << std::endl;
    return {vectors, graph_edges, static_cast<int>(ep_)};
}

#endif // SIMJOIN_COMMON_H