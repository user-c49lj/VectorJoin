#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <set>
#include <cmath>
#include <string>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <algorithm>
#include <sstream>
#include <atomic>
#include <filesystem>
#include <map>
#include <signal.h>
#include <sys/resource.h>

// Shared common implementations
#include "ws_common.h"

bool print_join_output = false;

// Thread-local distance computation counter (definition for simple_simjoin)
thread_local long long thread_distance_computations = 0;

// Function to get current memory usage in MB
double get_memory_usage_mb() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    return static_cast<double>(r_usage.ru_maxrss) / 1024.0; // Convert KB to MB
}

// Signal handling for graceful shutdown
volatile bool g_shutdown_requested = false;

// Profiling structure
struct ProfileData {
    std::map<std::string, std::chrono::nanoseconds> timings;
    std::map<std::string, int> call_counts;
    std::mutex mtx;
    
    void add_timing(const std::string& name, std::chrono::nanoseconds duration) {
        std::lock_guard<std::mutex> lock(mtx);
        timings[name] += duration;
        call_counts[name]++;
    }
    
    void print_stats() {
        std::lock_guard<std::mutex> lock(mtx);
        std::cout << "\n=== PROFILING STATISTICS ===" << std::endl;
        for (const auto& [name, duration] : timings) {
            auto count = call_counts[name];
            auto avg_duration = duration / count;
            std::cout << name << ": " 
                      << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms total, "
                      << count << " calls, "
                      << std::chrono::duration_cast<std::chrono::microseconds>(avg_duration).count() << "μs avg" << std::endl;
        }
    }
    
    void save_stats_to_file(const std::string& filename) {
        std::lock_guard<std::mutex> lock(mtx);
        std::ofstream file(filename);
        if (file.is_open()) {
            file << "=== PROFILING STATISTICS ===" << std::endl;
            file << "Timestamp: " << std::chrono::system_clock::now().time_since_epoch().count() << std::endl;
            file << "Program interrupted: " << (g_shutdown_requested ? "Yes" : "No") << std::endl;
            file << std::endl;
            
            for (const auto& [name, duration] : timings) {
                auto count = call_counts[name];
                auto avg_duration = duration / count;
                file << name << ": " 
                     << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() << "ms total, "
                     << count << " calls, "
                     << std::chrono::duration_cast<std::chrono::microseconds>(avg_duration).count() << "μs avg" << std::endl;
            }
            file.close();
            std::cout << "Profiling results saved to: " << filename << std::endl;
        } else {
            std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        }
    }
};

// Global profiling data
ProfileData g_profile_data;

void signal_handler(int signal) {
    std::cout << "\nReceived signal " << signal << ", shutting down gracefully..." << std::endl;
    g_shutdown_requested = true;
}

// Profiling macro
#define PROFILE_SCOPE(name) \
    auto start_##name = std::chrono::high_resolution_clock::now(); \
    struct ProfileScope_##name { \
        std::chrono::high_resolution_clock::time_point start; \
        std::string name; \
        ProfileScope_##name(const std::string& n) : start(std::chrono::high_resolution_clock::now()), name(n) {} \
        ~ProfileScope_##name() { \
            auto end = std::chrono::high_resolution_clock::now(); \
            g_profile_data.add_timing(name, end - start); \
        } \
    }; \
    ProfileScope_##name profile_scope_##name(#name)

// l2_distance is now defined in ws_common.h


// Function to count vectors in fvecs file
int count_vectors_in_fvecs(const std::string& filename, int dimension) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    int count = 0;
    while (file.good()) {
        // Read dimension from fvecs format (4-byte integer)
        int32_t vec_dim;
        file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
        
        if (file.eof() || file.fail()) {
            break;
        }
        
        // Verify dimension matches expected dimension
        if (vec_dim != dimension) {
            std::cout << "Warning: Vector " << count << " has dimension " << vec_dim << ", expected " << dimension << std::endl;
        }
        
        // Read and discard vector data
        std::vector<float> temp_vector(vec_dim);
        file.read(reinterpret_cast<char*>(temp_vector.data()), vec_dim * sizeof(float));
        
        if (file.gcount() == vec_dim * sizeof(float)) {
            count++;
        } else {
            std::cout << "Reached end of file while reading vector " << count << " (expected " << vec_dim * sizeof(float) << " bytes, got " << file.gcount() << ")" << std::endl;
            break;
        }
    }
    
    file.close();
    std::cout << "Found " << count << " vectors in fvecs file: " << filename << std::endl;
    return count;
}

// Function to load embedding vectors from fvecs file
std::unordered_map<int, std::vector<float>> load_embedding_vectors(const std::string& filename, int num_vectors, int dimension) {
    std::unordered_map<int, std::vector<float>> vectors;
    
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::cout << "Loading " << num_vectors << " vectors of dimension " << dimension << " from fvecs file: " << filename << std::endl;
    
    int i = 0;
    while (num_vectors == -1 || i < num_vectors) {
        // Read dimension from fvecs format (4-byte integer)
        int32_t vec_dim;
        file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
        
        if (file.eof()) {
            std::cout << "Reached end of file after " << i << " vectors" << std::endl;
            break;
        }
        
        // Verify dimension matches expected dimension
        if (vec_dim != dimension) {
            std::cout << "Warning: Vector " << i << " has dimension " << vec_dim << ", expected " << dimension << std::endl;
        }
        
        // Read vector data
        std::vector<float> vector_data(vec_dim);
        file.read(reinterpret_cast<char*>(vector_data.data()), vec_dim * sizeof(float));
        
        if (file.gcount() == vec_dim * sizeof(float)) {
            vectors[i] = vector_data;
        } else {
            std::cout << "Reached end of file while reading vector " << i << std::endl;
            break;
        }
        
        i++;  // Increment counter
    }
    
    file.close();
    std::cout << "Successfully loaded " << vectors.size() << " vectors from fvecs file" << std::endl;
    return vectors;
}

// Function to load cluster data
struct ClusterData {
    std::vector<std::vector<int>> clusters;  // clusters[i] contains vector indices for cluster i
    std::vector<std::vector<float>> centroids;  // centroids[i] is the centroid of cluster i
    std::vector<std::vector<double>> centroid_distances;  // centroid_distances[i][j] is distance from vector j to centroid i
    std::unordered_map<int, std::pair<int, int>> vector_to_cluster;  // vector_id -> (cluster_id, position_in_cluster)
    
    // Anchor data for additional triangle inequality pruning
    std::vector<std::vector<std::vector<float>>> anchors;  // anchors[cluster_id][anchor_idx] is the anchor_idx-th anchor vector in cluster cluster_id
    std::vector<std::vector<std::vector<double>>> anchor_distances;  // anchor_distances[cluster_id][anchor_idx][vector_pos] is distance from vector at position vector_pos in cluster cluster_id to anchor anchor_idx
    bool has_anchors = false;  // Flag to indicate if anchor data is available
    
    // Shuffled data for sequential access optimization
    std::vector<std::vector<std::vector<float>>> shuffled_vectors;  // shuffled_vectors[cluster_id][pos] is the vector at position pos in cluster cluster_id
    std::vector<int> cluster_sizes;  // cluster_sizes[cluster_id] is the number of vectors in cluster cluster_id
    int vector_dimension = 0;  // Dimension of vectors
};

ClusterData load_cluster_data(const std::string& clusters_file, const std::string& centroids_file, 
                             const std::string& centroid_distances_file, bool load_anchors = false,
                             const std::string& anchors_file = "", const std::string& anchor_distances_file = "") {
    ClusterData data;
    
    // Load clusters
    std::ifstream clusters_stream(clusters_file);
    if (!clusters_stream.is_open()) {
        throw std::runtime_error("Could not open clusters file: " + clusters_file);
    }
    
    std::string line;
    int cluster_id = 0;
    while (std::getline(clusters_stream, line)) {
        std::istringstream iss(line);
        std::vector<int> cluster;
        int vector_id;
        while (iss >> vector_id) {
            cluster.push_back(vector_id);
            data.vector_to_cluster[vector_id] = {cluster_id, cluster.size() - 1};
        }
        data.clusters.push_back(std::move(cluster));
        cluster_id++;
    }
    clusters_stream.close();
    
    // Load centroids from text file (not pickle)
    // We'll need to create a centroids.txt file from simple_cluster_binary.py
    std::ifstream centroids_stream(centroids_file);
    if (!centroids_stream.is_open()) {
        throw std::runtime_error("Could not open centroids file: " + centroids_file);
    }
    
    for (int i = 0; i < data.clusters.size(); ++i) {
        std::vector<float> centroid;
        std::string centroid_line;
        std::getline(centroids_stream, centroid_line);
        std::istringstream iss(centroid_line);
        float value;
        while (iss >> value) {
            centroid.push_back(value);
        }
        data.centroids.push_back(std::move(centroid));
    }
    centroids_stream.close();
    
    // Load centroid distances
    std::ifstream distances_stream(centroid_distances_file);
    if (!distances_stream.is_open()) {
        throw std::runtime_error("Could not open centroid distances file: " + centroid_distances_file);
    }
    
    for (int i = 0; i < data.clusters.size(); ++i) {
        std::vector<double> distances;
        std::string dist_line;
        std::getline(distances_stream, dist_line);
        std::istringstream iss(dist_line);
        double dist;
        while (iss >> dist) {
            distances.push_back(dist);
        }
        data.centroid_distances.push_back(std::move(distances));
    }
    distances_stream.close();
    
    // Load anchor data if requested
    if (load_anchors && !anchors_file.empty() && !anchor_distances_file.empty()) {
        // Load anchors (per cluster)
        std::ifstream anchors_stream(anchors_file);
        if (!anchors_stream.is_open()) {
            throw std::runtime_error("Could not open anchors file: " + anchors_file);
        }
        
        // Initialize anchors for each cluster
        data.anchors.resize(data.clusters.size());
        
        std::string line;
        int cluster_idx = 0;
        while (std::getline(anchors_stream, line) && cluster_idx < data.clusters.size()) {
            std::istringstream iss(line);
            std::vector<float> current_anchor;
            std::vector<std::vector<float>> cluster_anchors;
            
            // Each line contains N anchor vectors flattened into one line
            // We need to reconstruct the N anchor vectors from the flattened line
            float value;
            while (iss >> value) {
                current_anchor.push_back(value);
                
                // If we've read a complete anchor vector (assuming all vectors have the same dimension)
                if (current_anchor.size() == data.centroids[0].size()) {
                    cluster_anchors.push_back(current_anchor);
                    current_anchor.clear();
                }
            }
            
            data.anchors[cluster_idx] = std::move(cluster_anchors);
            cluster_idx++;
        }
        anchors_stream.close();
        
        // Load anchor distances (per cluster)
        std::ifstream anchor_dist_stream(anchor_distances_file);
        if (!anchor_dist_stream.is_open()) {
            throw std::runtime_error("Could not open anchor distances file: " + anchor_distances_file);
        }
        
        // Initialize anchor distances for each cluster
        data.anchor_distances.resize(data.clusters.size());
        
        cluster_idx = 0;
        while (std::getline(anchor_dist_stream, line) && cluster_idx < data.clusters.size()) {
            std::istringstream iss(line);
            std::vector<std::vector<double>> cluster_distances;
            
            // Each line contains N*C distances where N = anchors per cluster, C = vectors in cluster
            // First C distances are for first anchor, next C for second anchor, etc.
            int num_anchors = data.anchors[cluster_idx].size();
            int num_vectors = data.clusters[cluster_idx].size();
            
            // Initialize the 2D structure
            cluster_distances.resize(num_anchors);
            for (int anchor_idx = 0; anchor_idx < num_anchors; ++anchor_idx) {
                cluster_distances[anchor_idx].resize(num_vectors);
            }
            
            // Read distances and organize them properly
            int anchor_idx = 0;
            int vector_pos = 0;
            double value;
            while (iss >> value) {
                cluster_distances[anchor_idx][vector_pos] = value;
                vector_pos++;
                if (vector_pos >= num_vectors) {
                    vector_pos = 0;
                    anchor_idx++;
                }
            }
            
            data.anchor_distances[cluster_idx] = std::move(cluster_distances);
            cluster_idx++;
        }
        anchor_dist_stream.close();
        
        data.has_anchors = true;
        int total_anchors = 0;
        for (const auto& cluster_anchors : data.anchors) {
            total_anchors += cluster_anchors.size();
        }
        std::cout << "Loaded " << total_anchors << " total anchors across " 
                  << data.clusters.size() << " clusters" << std::endl;
    }
    
    std::cout << "Loaded " << data.clusters.size() << " clusters with " 
              << data.centroids.size() << " centroids and " 
              << data.centroid_distances.size() << " distance arrays" << std::endl;
    
    return data;
}

// Function to create shuffled Y_vectors for sequential access optimization
void create_shuffled_vectors(ClusterData& cluster_data, const std::unordered_map<int, std::vector<float>>& Y_vectors) {
    std::cout << "Creating shuffled vectors for sequential access optimization..." << std::endl;
    
    // Determine vector dimension from the first vector
    if (Y_vectors.empty()) {
        throw std::runtime_error("No vectors available to shuffle");
    }
    cluster_data.vector_dimension = Y_vectors.begin()->second.size();
    
    // Initialize cluster data structures
    cluster_data.cluster_sizes.resize(cluster_data.clusters.size());
    cluster_data.shuffled_vectors.resize(cluster_data.clusters.size());
    
    for (size_t cluster_id = 0; cluster_id < cluster_data.clusters.size(); ++cluster_id) {
        const auto& cluster = cluster_data.clusters[cluster_id];
        cluster_data.cluster_sizes[cluster_id] = cluster.size();
        
        // Allocate space for all vectors in this cluster
        cluster_data.shuffled_vectors[cluster_id].resize(cluster.size());
        
        // Copy vectors as std::vector<float> for each position
        for (size_t pos = 0; pos < cluster.size(); ++pos) {
            int vector_id = cluster[pos];
            cluster_data.shuffled_vectors[cluster_id][pos] = Y_vectors.at(vector_id);
        }
    }
    
    std::cout << "Shuffled vectors created with dimension " << cluster_data.vector_dimension 
              << " across " << cluster_data.clusters.size() << " clusters" << std::endl;
}

// Function to load vectors from global fvecs file using cluster assignments
std::unordered_map<int, std::vector<float>> load_vectors_from_clusters(const std::string& global_fvecs_file,
                                                                      const ClusterData& cluster_data) {
    std::unordered_map<int, std::vector<float>> vectors;
    
    // Determine vector dimension from the first centroid
    int vector_dim = cluster_data.centroids[0].size();
    std::cout << "Detected vector dimension: " << vector_dim << " (from centroids)" << std::endl;
    
    // Open the global fvecs file
    std::ifstream file(global_fvecs_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open global fvecs file: " + global_fvecs_file);
    }
    
    std::cout << "Loading vectors from global fvecs file: " << global_fvecs_file << std::endl;
    
    // Create a set of all vector IDs that belong to any cluster
    std::set<int> all_cluster_vector_ids;
    for (const auto& cluster : cluster_data.clusters) {
        for (int vector_id : cluster) {
            all_cluster_vector_ids.insert(vector_id);
        }
    }
    
    std::cout << "Found " << all_cluster_vector_ids.size() << " unique vectors across all clusters" << std::endl;
    
    // Read vectors from the global fvecs file
    int vector_index = 0;
    while (!file.eof()) {
        // Read dimension from fvecs format (4-byte integer)
        int32_t vec_dim;
        file.read(reinterpret_cast<char*>(&vec_dim), sizeof(int32_t));
        
        if (file.eof()) {
            break;
        }
        
        // Verify dimension matches expected dimension
        if (vec_dim != vector_dim) {
            std::cout << "Warning: Vector " << vector_index << " has dimension " << vec_dim << ", expected " << vector_dim << std::endl;
        }
        
        // Read vector data
        std::vector<float> vector_data(vec_dim);
        file.read(reinterpret_cast<char*>(vector_data.data()), vec_dim * sizeof(float));
        
        if (file.gcount() == vec_dim * sizeof(float)) {
            // Only store this vector if it belongs to any cluster
            if (all_cluster_vector_ids.find(vector_index) != all_cluster_vector_ids.end()) {
                vectors[vector_index] = vector_data;
            }
        } else {
            std::cout << "Reached end of file while reading vector " << vector_index << std::endl;
            break;
        }
        
        vector_index++;
    }
    
    file.close();
    std::cout << "Loaded " << vectors.size() << " vectors from global fvecs file using cluster assignments" << std::endl;
    return vectors;
}

// Function to process a chunk of X vectors with multi-cluster support
void process_x_chunk_multi_cluster(const std::vector<int>& x_chunk,
                                  const std::unordered_map<int, std::vector<float>>& X_vectors,
                                  const ClusterData& cluster_data,
                                  const std::unordered_map<int, std::vector<float>>& Y_vectors,
                                  double epsilon, int max_pairs,
                                  std::atomic<int>& processed_count, std::atomic<int>& total_pairs,
                                  std::atomic<long long>& total_distance_computations, std::atomic<long long>& total_pruned,
                                  std::mutex& progress_mutex, std::mutex& output_mutex,
                                  std::ofstream* streaming_file, const std::chrono::high_resolution_clock::time_point& start_time,
                                  int total_query_vectors, const std::string& result_dir) {
    
    PROFILE_SCOPE(process_x_chunk_multi_cluster);
    
    for (int x_id : x_chunk) {
        // Check for shutdown request
        if (g_shutdown_requested) {
            std::cout << "Shutdown requested, stopping processing..." << std::endl;
            break;
        }
        
        try {
            PROFILE_SCOPE(process_single_query_multi_cluster);
            const auto& x_vec = X_vectors.at(x_id);
            long long distance_count = 0;
            int pairs_found_this_query = 0;
            long long pruned_count = 0;  // Count of vectors pruned by triangle inequality
            
            // Start timing for this specific query
            auto query_start_time = std::chrono::high_resolution_clock::now();
            
            // Collect matches for this query
            std::vector<std::pair<int, double>> matches;
            
            // Process each cluster
            for (size_t cluster_id = 0; cluster_id < cluster_data.clusters.size(); ++cluster_id) {
                const auto& cluster = cluster_data.clusters[cluster_id];
                const auto& centroid = cluster_data.centroids[cluster_id];
                const auto& centroid_dists = cluster_data.centroid_distances[cluster_id];

                // Compute distance from query to centroid
                double xc_distance = l2_distance(x_vec, centroid);
                
                // Compute distances from query to all anchors in this cluster (if anchors are available)
                std::vector<double> xa_distances;
                if (cluster_data.has_anchors) {
                    xa_distances.reserve(cluster_data.anchors[cluster_id].size());
                    for (size_t anchor_idx = 0; anchor_idx < cluster_data.anchors[cluster_id].size(); ++anchor_idx) {
                        double xa_distance = l2_distance(x_vec, cluster_data.anchors[cluster_id][anchor_idx]);
                        xa_distances.push_back(xa_distance);
                    }
                }
                
                // Process each vector in this cluster using sequential access
                for (size_t pos = 0; pos < cluster.size(); ++pos) {
                    int y_id = cluster[pos];
                    
                    // Get vector data from shuffled storage for sequential access
                    const auto& y_vec = cluster_data.shuffled_vectors[cluster_id][pos];
                    
                    // Get distance from vector to its centroid
                    double vc_distance = centroid_dists[pos];
                    
                    {
                        //PROFILE_SCOPE(triangle_inequality_pruning); XXX this makes the program much slower
                        // Apply triangle inequality pruning rule
                        if (std::abs(xc_distance - vc_distance) >= epsilon) {
                            // From triangle inequality: |distance(x,v) - distance(v,c)| >= threshold
                            // This means distance(x,v) >= threshold, so we can safely prune v
                            pruned_count++;
                            continue;
                        }
                        else if (xc_distance + vc_distance <= epsilon) {
                            // We know that distance(x, v) < distance(x,v) + distance(v,c) <= epsilon, so we can safely prune v
                            pruned_count++;
                            matches.push_back({y_id, xc_distance + vc_distance});
                            pairs_found_this_query++;
                            if (pairs_found_this_query >= max_pairs) break;
                            continue;
                        }
                        
                        // Apply anchor-based triangle inequality pruning if anchors are available
                        if (cluster_data.has_anchors) {
                            //PROFILE_SCOPE(anchor_triangle_inequality_pruning);
                            bool do_continue = false;
                            for (size_t anchor_idx = 0; anchor_idx < cluster_data.anchors[cluster_id].size(); ++anchor_idx) {
                                // Use precomputed distance from query x to anchor a in this cluster
                                double xa_distance = xa_distances[anchor_idx];
                                
                                // Get distance from vector v to anchor a (from anchor_distances.txt)
                                // anchor_distances[cluster_id][anchor_idx][pos] where pos is the position of vector y_id in cluster cluster_id
                                double va_distance = cluster_data.anchor_distances[cluster_id][anchor_idx][pos];
                                
                                // Apply triangle inequality: |xa_distance - va_distance| >= epsilon
                                if (std::abs(xa_distance - va_distance) >= epsilon) {
                                    do_continue = true;
                                    break;
                                }
                                else if (xa_distance + va_distance <= epsilon) {
                                    do_continue = true;
                                    matches.push_back({y_id, xa_distance + va_distance});
                                    pairs_found_this_query++;
                                    break;
                                }
                            }
                            if (do_continue) {
                                pruned_count++;
                                continue; // to next v
                            }
                        }
                    }
                    
                    // Fall back to computing actual distance using sequential data
                    double distance = l2_distance(x_vec, y_vec);
                    distance_count++;
                    
                    if (distance <= epsilon) {
                        matches.push_back({y_id, distance});
                        pairs_found_this_query++;
                        if (pairs_found_this_query >= max_pairs) break;
                    }
                }
                
                if (pairs_found_this_query >= max_pairs) break;
            }
            
            // End timing for this specific query
            auto query_end_time = std::chrono::high_resolution_clock::now();
            auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end_time - query_start_time);
            
            // Write streaming output for this query (per-query time, not accumulated)
            if (streaming_file) {
                std::lock_guard<std::mutex> lock(output_mutex);
                double memory_usage = get_memory_usage_mb();
                *streaming_file << x_id << " " << matches.size() << " " << static_cast<double>(distance_count) 
                               << " " << static_cast<double>(query_duration.count()) / 1000.0 << " " << memory_usage
                               << " " << pruned_count;  // Add pruned count to output
                if (print_join_output) { // XXX don't print outputs to save disk space
                    for (const auto& [y_id, distance] : matches) {
                        *streaming_file << " " << y_id;
                    }
                }
                *streaming_file << std::endl;
                streaming_file->flush();
            }
            
            // Update progress
            int current_processed = processed_count.fetch_add(1) + 1;
            int current_pairs = total_pairs.fetch_add(pairs_found_this_query) + pairs_found_this_query;
            total_distance_computations.fetch_add(distance_count);
            total_pruned.fetch_add(pruned_count);
            
            // Print progress every 100 processed queries
            if (current_processed % 100 == 0) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                long long current_total_distance = total_distance_computations.load();
                long long current_total_pruned = total_pruned.load();
                std::cout << "Progress: " << current_processed << "/" << total_query_vectors 
                         << " queries processed, " << current_pairs << " pairs found, "
                         << current_total_distance << " total distance computations, "
                         << current_total_pruned << " total vectors pruned, "
                         << elapsed.count() << " ms elapsed" << std::endl;
                
                // Stream progress to file
                static std::ofstream progress_file(result_dir + "/progress.txt");
                if (progress_file.is_open()) {
                    double memory_usage = get_memory_usage_mb();
                    progress_file << current_processed << " " << current_pairs 
                                 << " " << current_total_distance << " " 
                                 << static_cast<double>(elapsed.count()) << " " << memory_usage << std::endl;
                    progress_file.flush();
                }
            }
        } catch (const std::out_of_range&) {
            // Skip if x_id not found in X_vectors
            continue;
        }
    }
}

// Function to process a chunk of X vectors (original single-file mode)
void process_x_chunk(const std::vector<int>& x_chunk,
                     const std::unordered_map<int, std::vector<float>>& X_vectors,
                     const std::unordered_map<int, std::vector<float>>& Y_vectors,
                     double epsilon, int max_pairs,
                     std::atomic<int>& processed_count, std::atomic<int>& total_pairs, std::atomic<long long>& total_distance_computations,
                     std::mutex& progress_mutex, std::mutex& output_mutex,
                     std::ofstream* streaming_file, const std::chrono::high_resolution_clock::time_point& start_time,
                     int total_query_vectors, const std::string& result_dir) {
    
    PROFILE_SCOPE(process_x_chunk);
    
    for (int x_id : x_chunk) {
        // Check for shutdown request
        if (g_shutdown_requested) {
            std::cout << "Shutdown requested, stopping processing..." << std::endl;
            break;
        }
        
        try {
            PROFILE_SCOPE(process_single_query);
            const auto& x_vec = X_vectors.at(x_id);
            long long distance_count = 0;
            int pairs_found_this_query = 0;
            
            // Start timing for this specific query
            auto query_start_time = std::chrono::high_resolution_clock::now();
            
            // Collect matches for this query
            std::vector<std::pair<int, double>> matches;
            
            for (const auto& [y_id, y_vec] : Y_vectors) {
                double distance = l2_distance(x_vec, y_vec);
                distance_count++;
                
                if (distance <= epsilon) {
                    matches.push_back({y_id, distance});
                    pairs_found_this_query++;
                    if (pairs_found_this_query >= max_pairs) break;
                }
            }
            
            // End timing for this specific query
            auto query_end_time = std::chrono::high_resolution_clock::now();
            auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end_time - query_start_time);
            
            // Write streaming output for this query (per-query time, not accumulated)
            if (streaming_file) {
                std::lock_guard<std::mutex> lock(output_mutex);
                double memory_usage = get_memory_usage_mb();
                *streaming_file << x_id << " " << matches.size() << " " << static_cast<double>(distance_count) 
                               << " " << static_cast<double>(query_duration.count()) / 1000.0 << " " << memory_usage;
                if (print_join_output) { // XXX don't print outputs to save disk space
                    for (const auto& [y_id, distance] : matches) {
                        *streaming_file << " " << y_id;
                    }
                }
                *streaming_file << std::endl;
                streaming_file->flush();
            }
            
            // Update progress
            int current_processed = processed_count.fetch_add(1) + 1;
            int current_pairs = total_pairs.fetch_add(pairs_found_this_query) + pairs_found_this_query;
            total_distance_computations.fetch_add(distance_count);
            
            // Print progress every 100 processed queries
            if (current_processed % 100 == 0) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                long long current_total_distance = total_distance_computations.load();
                std::cout << "Progress: " << current_processed << "/" << total_query_vectors 
                         << " queries processed, " << current_pairs << " pairs found, "
                         << current_total_distance << " total distance computations, "
                         << elapsed.count() << " ms elapsed" << std::endl;
                
                // Stream progress to file
                static std::ofstream progress_file(result_dir + "/progress.txt");
                if (progress_file.is_open()) {
                    double memory_usage = get_memory_usage_mb();
                    progress_file << current_processed << " " << current_pairs 
                                 << " " << current_total_distance << " " 
                                 << static_cast<double>(elapsed.count()) << " " << memory_usage << std::endl;
                    progress_file.flush();
                }
            }
        } catch (const std::out_of_range&) {
            // Skip if x_id not found in X_vectors
            continue;
        }
    }
}

// Main similarity join function with parallelization (single-file mode)
void simple_similarity_join(const std::unordered_map<int, std::vector<float>>& X_vectors,
                      const std::unordered_map<int, std::vector<float>>& Y_vectors,
                      double epsilon = 0.5, int max_pairs = 10000, int num_threads = 1,
                      std::ofstream* streaming_file = nullptr,
                      const std::string& result_dir = "") {
    
    PROFILE_SCOPE(simple_similarity_join);
    std::cout << "Running simple similarity join with " << num_threads << " threads..." << std::endl;
    
    // Progress tracking variables
    std::atomic<int> processed_count(0);
    std::atomic<int> total_pairs(0);
    std::atomic<long long> total_distance_computations(0);
    std::mutex progress_mutex;
    std::mutex output_mutex;
    auto start_time = std::chrono::high_resolution_clock::now();
    int total_query_vectors = X_vectors.size();
    
    // Convert X_vectors keys to vector for chunking
    std::vector<int> x_ids;
    for (const auto& [id, _] : X_vectors) {
        x_ids.push_back(id);
    }
    
    if (x_ids.empty()) {
        return;
    }
    
    // Calculate chunk size
    int chunk_size = (x_ids.size() + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    
    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        // Check for shutdown request
        if (g_shutdown_requested) {
            std::cout << "Shutdown requested, stopping thread launch..." << std::endl;
            break;
        }
        
        int start_idx = i * chunk_size;
        int end_idx = std::min(start_idx + chunk_size, static_cast<int>(x_ids.size()));
        
        if (start_idx < x_ids.size()) {
            std::vector<int> chunk(x_ids.begin() + start_idx, x_ids.begin() + end_idx);
            threads.emplace_back([&, i, chunk]() {
                process_x_chunk(chunk, X_vectors, Y_vectors, epsilon, max_pairs,
                                                   processed_count, total_pairs, total_distance_computations, progress_mutex, output_mutex, 
                                                   streaming_file, start_time, total_query_vectors, result_dir);
            });
        }
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "All threads completed" << std::endl;
}

// Main similarity join function with parallelization (multi-cluster mode)
void simple_similarity_join_multi_cluster(const std::unordered_map<int, std::vector<float>>& X_vectors,
                                         const ClusterData& cluster_data,
                                         const std::unordered_map<int, std::vector<float>>& Y_vectors,
                                         double epsilon = 0.5, int max_pairs = 10000, int num_threads = 1,
                                         std::ofstream* streaming_file = nullptr,
                                         const std::string& result_dir = "") {
    
    PROFILE_SCOPE(simple_similarity_join_multi_cluster);
    std::cout << "Running multi-cluster similarity join with " << num_threads << " threads..." << std::endl;
    std::cout << "Using triangle inequality pruning with " << cluster_data.clusters.size() << " clusters" << std::endl;
    if (cluster_data.has_anchors) {
        int total_anchors = 0;
        for (const auto& cluster_anchors : cluster_data.anchors) {
            total_anchors += cluster_anchors.size();
        }
        std::cout << "Using anchor-based triangle inequality pruning with " << total_anchors << " total anchors across " 
                  << cluster_data.clusters.size() << " clusters" << std::endl;
    }
    
    // Progress tracking variables
    std::atomic<int> processed_count(0);
    std::atomic<int> total_pairs(0);
    std::atomic<long long> total_distance_computations(0);
    std::atomic<long long> total_pruned(0);
    std::mutex progress_mutex;
    std::mutex output_mutex;
    auto start_time = std::chrono::high_resolution_clock::now();
    int total_query_vectors = X_vectors.size();
    
    // Convert X_vectors keys to vector for chunking
    std::vector<int> x_ids;
    for (const auto& [id, _] : X_vectors) {
        x_ids.push_back(id);
    }
    
    if (x_ids.empty()) {
        return;
    }
    
    // Calculate chunk size
    int chunk_size = (x_ids.size() + num_threads - 1) / num_threads;
    
    std::vector<std::thread> threads;
    
    // Launch threads
    for (int i = 0; i < num_threads; ++i) {
        // Check for shutdown request
        if (g_shutdown_requested) {
            std::cout << "Shutdown requested, stopping thread launch..." << std::endl;
            break;
        }
        
        int start_idx = i * chunk_size;
        int end_idx = std::min(start_idx + chunk_size, static_cast<int>(x_ids.size()));
        
        if (start_idx < x_ids.size()) {
            std::vector<int> chunk(x_ids.begin() + start_idx, x_ids.begin() + end_idx);
            threads.emplace_back([&, i, chunk]() {
                process_x_chunk_multi_cluster(chunk, X_vectors, cluster_data, Y_vectors, epsilon, max_pairs,
                                            processed_count, total_pairs, total_distance_computations, total_pruned, progress_mutex, output_mutex, 
                                            streaming_file, start_time, total_query_vectors, result_dir);
            });
        }
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "All threads completed" << std::endl;
}

int main(int argc, char* argv[]) {
    // Set up signal handling
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    if (argc < 9) {
        std::cout << "Usage: " << argv[0] << " <result_dir> <epsilon> <num_threads> <Q> <supplier_embedding_path> <part_embedding_path> <prefix> <dimension> [cluster_dir] [use_anchors]" << std::endl;
        std::cout << "  result_dir: result directory name (or 'auto' to generate automatically)" << std::endl;
        std::cout << "\nSingle-file mode (default):" << std::endl;
        std::cout << "  <result_dir> <epsilon> <num_threads> <Q> <supplier_embedding_path> <part_embedding_path> <prefix> <dimension>" << std::endl;
        std::cout << "\nMulti-cluster mode:" << std::endl;
        std::cout << "  <result_dir> <epsilon> <num_threads> <Q> <supplier_embedding_path> <part_embedding_path> <prefix> <dimension> <cluster_dir>" << std::endl;
        std::cout << "\nMulti-cluster mode with anchor pruning:" << std::endl;
        std::cout << "  <result_dir> <epsilon> <num_threads> <Q> <supplier_embedding_path> <part_embedding_path> <prefix> <dimension> <cluster_dir> <use_anchors>" << std::endl;
        std::cout << "\nMulti-cluster mode enables triangle inequality pruning for faster similarity search." << std::endl;
        std::cout << "Anchor pruning adds additional triangle inequality pruning using anchor points." << std::endl;
        std::cout << "\nNote: In multi-cluster mode, the following files must exist in <cluster_dir>:" << std::endl;
        std::cout << "  - clusters.txt (cluster assignments)" << std::endl;
        std::cout << "  - centroids.txt (centroid coordinates)" << std::endl;
        std::cout << "  - centroid_distances.txt (distances to centroids)" << std::endl;
        std::cout << "  - cluster_*.bin files (vector data)" << std::endl;
        std::cout << "\nNote: When anchor pruning is enabled, the following files must also exist:" << std::endl;
        std::cout << "  - anchors.txt (anchor coordinates)" << std::endl;
        std::cout << "  - anchor_distances.txt (distances from vectors to anchors)" << std::endl;
        return 1;
    }
    
    // Parse command line arguments
    std::string result_dir_arg = argv[1];
    double EPSILON = std::stod(argv[2]);
    int NUM_THREADS = std::stoi(argv[3]);
    int Q = std::stoi(argv[4]);
    std::string SUPPLIER_EMBEDDING_PATH = argv[5];
    std::string PART_EMBEDDING_PATH = argv[6];
    std::string PREFIX = argv[7];
    int DIMENSION = std::stoi(argv[8]);
    
    std::cout << "=== Simple Supplier-Part SimJoin (C++) ===" << std::endl;
    std::cout << "Parameters: epsilon=" << EPSILON << ", threads=" << NUM_THREADS << std::endl;
    
    // Check if multi-cluster mode is enabled
    bool multi_cluster_mode = (argc >= 9) && (std::string(argv[8]) == "true" || std::string(argv[8]) == "1" || std::string(argv[8]) == "yes");
    bool use_anchors = false;
    
    if (multi_cluster_mode) {
        std::cout << "Multi-cluster mode enabled with triangle inequality pruning" << std::endl;
        
        // Check if anchor pruning is enabled
        if (argc >= 10) {
            std::string anchor_param = argv[9];
            use_anchors = (anchor_param == "true" || anchor_param == "1" || anchor_param == "yes");
            if (use_anchors) {
                std::cout << "Anchor-based triangle inequality pruning enabled" << std::endl;
            }
        }
    } else {
        std::cout << "Single-file mode (no clustering)" << std::endl;
    }
    
    try {
        PROFILE_SCOPE(main_total);
        
        // Configuration
        const int SUPPLIER_DIM = DIMENSION;
        const int PART_DIM = DIMENSION;
        const int SUPPLIER_COUNT = Q;   // Use Q parameter to limit query vectors
        
        // Count actual number of vectors in data fvecs file
        int PART_COUNT = count_vectors_in_fvecs(PART_EMBEDDING_PATH, PART_DIM);
        
        // Load embedding vectors
        {
            PROFILE_SCOPE(load_vectors);
            auto supplier_vectors = load_embedding_vectors(SUPPLIER_EMBEDDING_PATH, SUPPLIER_COUNT, SUPPLIER_DIM);
            auto part_vectors = load_embedding_vectors(PART_EMBEDDING_PATH, PART_COUNT, PART_DIM);
            
            std::cout << "\nData loaded:" << std::endl;
            std::cout << "Supplier vectors: " << supplier_vectors.size() << std::endl;
            std::cout << "Part vectors: " << part_vectors.size() << std::endl;
            
            // Create output directory
            std::string result_dir;
            if (result_dir_arg == "auto") {
                // Generate directory name automatically
                result_dir = "simple_result/" + PREFIX + "_epsilon_" + std::to_string(EPSILON) + "_threads_" + std::to_string(NUM_THREADS) + "_Q" + std::to_string(Q);
                if (multi_cluster_mode) {
                    result_dir += "_multicluster";
                    if (use_anchors) {
                        result_dir += "_anchors";
                    }
                }
            } else {
                // Use provided directory name
                result_dir = result_dir_arg;
            }
            std::filesystem::create_directories(result_dir);
            
            // Create streaming output file
            std::string streaming_filename = result_dir + "/join_output.txt";
            std::ofstream streaming_file(streaming_filename);
            if (!streaming_file.is_open()) {
                std::cerr << "Error: Could not open streaming output file " << streaming_filename << std::endl;
                return 1;
            }
            
            // Run similarity join with streaming output
            std::cout << "\nRunning simple simjoin with epsilon=" << EPSILON << "..." << std::endl;
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            if (multi_cluster_mode) {
                // Multi-cluster mode
                std::string cluster_dir = argv[9];
                
                // Automatically find required files in the cluster directory
                std::string clusters_file = cluster_dir + "/clusters.txt";
                std::string centroids_file = cluster_dir + "/centroids.txt";
                std::string centroid_distances_file = cluster_dir + "/centroid_distances.txt";
                
                std::cout << "Multi-cluster mode: Looking for files in " << cluster_dir << std::endl;
                std::cout << "  - clusters.txt: " << clusters_file << std::endl;
                std::cout << "  - centroids.txt: " << centroids_file << std::endl;
                std::cout << "  - centroid_distances.txt: " << centroid_distances_file << std::endl;
                
                // Verify all required files exist
                if (!std::filesystem::exists(clusters_file)) {
                    throw std::runtime_error("clusters.txt not found in cluster directory: " + cluster_dir);
                }
                if (!std::filesystem::exists(centroids_file)) {
                    throw std::runtime_error("centroids.txt not found in cluster directory: " + cluster_dir);
                }
                if (!std::filesystem::exists(centroid_distances_file)) {
                    throw std::runtime_error("centroid_distances.txt not found in cluster directory: " + cluster_dir);
                }
                
                // Note: We now read from the global fvecs file using cluster assignments
                // No need to check for individual cluster binary files
                
                // Check for anchor files if anchor pruning is enabled
                std::string anchors_file = cluster_dir + "/anchors.txt";
                std::string anchor_distances_file = cluster_dir + "/anchor_distances.txt";
                
                if (use_anchors) {
                    std::cout << "Anchor pruning enabled: Looking for anchor files in " << cluster_dir << std::endl;
                    std::cout << "  - anchors.txt: " << anchors_file << std::endl;
                    std::cout << "  - anchor_distances.txt: " << anchor_distances_file << std::endl;
                    
                    if (!std::filesystem::exists(anchors_file)) {
                        throw std::runtime_error("anchors.txt not found in cluster directory: " + cluster_dir);
                    }
                    if (!std::filesystem::exists(anchor_distances_file)) {
                        throw std::runtime_error("anchor_distances.txt not found in cluster directory: " + cluster_dir);
                    }
                }
                
                std::cout << "Loading cluster data..." << std::endl;
                auto cluster_data = load_cluster_data(clusters_file, centroids_file, centroid_distances_file, 
                                                    use_anchors, anchors_file, anchor_distances_file);
                
                // Load vectors from global fvecs file using cluster assignments
                auto cluster_vectors = load_vectors_from_clusters(PART_EMBEDDING_PATH, cluster_data);
                
                // Create shuffled vectors for sequential access optimization
                create_shuffled_vectors(cluster_data, cluster_vectors);
                
                if (use_anchors) {
                    std::cout << "Running multi-cluster similarity join with anchor-based triangle inequality pruning..." << std::endl;
                } else {
                    std::cout << "Running multi-cluster similarity join..." << std::endl;
                }
                simple_similarity_join_multi_cluster(supplier_vectors, cluster_data, cluster_vectors, 
                                                   EPSILON, PART_COUNT, NUM_THREADS, &streaming_file, result_dir);
            } else {
                // Single-file mode (original)
                std::cout << "Running single-file similarity join..." << std::endl;
                simple_similarity_join(supplier_vectors, part_vectors, EPSILON, PART_COUNT, NUM_THREADS, &streaming_file, result_dir);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto join_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
            
            streaming_file.close();
            
            // Get final counts from the streaming file to show summary
            std::ifstream count_file(streaming_filename);
            int total_pairs_found = 0;
            int total_queries_processed = 0;
            long long total_distance_computations = 0;
            long long total_pruned = 0;
            std::string line;
            while (std::getline(count_file, line)) {
                std::istringstream iss(line);
                int x_id, num_matches;
                double distance_count, processing_time, memory_usage;
                iss >> x_id >> num_matches >> distance_count >> processing_time >> memory_usage;
                
                total_pairs_found += num_matches;
                total_queries_processed++;
                total_distance_computations += static_cast<int>(distance_count);
                
                // Check if pruned count is present (multi-cluster mode)
                if (multi_cluster_mode) {
                    int query_pruned_count;
                    if (iss >> query_pruned_count) {
                        total_pruned += query_pruned_count;
                    }
                }
            }
            count_file.close();
            
            std::cout << "\nResults:" << std::endl;
            std::cout << "Join time: " << join_time << " seconds" << std::endl;
            std::cout << "Total queries processed: " << total_queries_processed << std::endl;
            std::cout << "Total pairs found: " << total_pairs_found << std::endl;
            std::cout << "Total distance computations: " << total_distance_computations << std::endl;
            if (multi_cluster_mode) {
                std::cout << "Total vectors pruned by triangle inequality: " << total_pruned << std::endl;
                std::cout << "Pruning efficiency: " << std::fixed << std::setprecision(2) 
                         << (100.0 * total_pruned / (total_distance_computations + total_pruned)) << "%" << std::endl;
            }
            
            std::cout << "Streaming results saved to " << streaming_filename << std::endl;
            
            // Check if shutdown was requested
            if (g_shutdown_requested) {
                std::cout << "\nProgram was interrupted by user request." << std::endl;
            }
            
            // Print profiling statistics
            g_profile_data.print_stats();
            
            // Save profiling results to file
            std::string profile_filename = result_dir + "/profile_stats.txt";
            g_profile_data.save_stats_to_file(profile_filename);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
