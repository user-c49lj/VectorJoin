#include <iostream>
#include <vector>
#include <limits>
#include <sstream>
#include <tuple>
#include <unordered_set>
#include <iomanip>

// Profiling includes
#include <map>
#include <string>
#include <chrono>
#include <mutex>
#include <fstream>
#include <signal.h>
#include <sys/resource.h>
#include <unordered_map>
#include <filesystem>

//#include <boost/stacktrace.hpp>

// Shared common implementations
#include "ws_common.h"

bool print_join_output = false;

// Function to get current memory usage in MB
double get_memory_usage_mb() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
    return static_cast<double>(r_usage.ru_maxrss) / 1024.0; // Convert KB to MB
}
#include <atomic>

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

// Global distance computation counters
std::atomic<long long> g_distance_computations{0};
std::atomic<long long> g_seed_distance_computations{0};
std::atomic<long long> g_neighbor_distance_computations{0};
std::atomic<long long> g_bfs_distance_computations{0};

// Global query-to-data edges counter
std::atomic<long long> g_query_to_data_edges{0};
std::atomic<long long> g_query_to_data_edges_traversed{0};

// Global data-to-data edges counter
std::atomic<long long> g_data_to_data_edges{0};
std::atomic<long long> g_data_to_data_edges_traversed{0};

// Global query-to-query edges counter
std::atomic<long long> g_query_to_query_edges{0};
std::atomic<long long> g_query_to_query_edges_traversed{0};

// Global data structures for BFS collection
std::vector<std::vector<int>> g_bfs_seeds;  // g_bfs_seeds[query_id] = list of seed nodes
std::vector<std::vector<std::pair<int, int>>> g_bfs_visited;  // g_bfs_visited[query_id] = list of (node, depth) pairs

// Global data structures for greedy search collection
std::vector<std::vector<std::pair<int, int>>> g_greedy_visited;  // g_greedy_visited[query_id] = list of (node, depth) pairs visited during greedy search
std::vector<std::vector<int>> g_final_join_results;  // g_final_join_results[query_id] = final J_i after pruning with epsilon

// Triangle inequality data structure
struct TriangleInequalityData {
    std::vector<std::vector<float>> centroids;
    std::vector<std::vector<double>> centroid_distances;
    std::unordered_map<int, std::pair<int, int>> vector_to_cluster;
};

// Clustering structures
struct ClusterIndex {
    std::string cluster_file;                    // Path to cluster_X.bin file
    std::string cagra_file;                      // Path to cluster_X.nsg.cagra file
    std::unordered_map<int, int> local_to_global_id;  // Maps local ID j to global ID i
    std::unordered_map<int, std::vector<float>> vectors;  // Cluster vectors
    std::unordered_map<int, std::vector<int>> neighbors;  // CAGRA neighbors
    int cluster_id;                              // Cluster identifier
    int num_vectors;                             // Number of vectors in this cluster
};

// Global clustering data
std::vector<ClusterIndex> g_cluster_indexes;

// Global variables for multi-cluster mode
std::vector<std::unordered_map<int, std::unordered_set<int>>> g_cluster_windows;
std::vector<std::unordered_map<int, double>> g_cluster_processing_times;
std::unordered_map<int, std::pair<int, int>> g_global_to_cluster;  // Maps global ID i to (cluster_id, local_id)

// Thread-local distance computation counter
thread_local long long thread_distance_computations = 0;

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
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <deque>
#include <stack>
#include <algorithm>
#include <fstream>

#include <chrono>
#include <cmath>
#include <random>
#include <set>

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <limits>
#include <filesystem>
#include <cassert>
#include "json_wrapper.h"
#include <climits>

std::vector<bool> g_is_node_ood; // Global storage for OOD status
int id_matches = 0;
int ood_matches = 0;


// ----------------------------------------------------------------------
// Algorithm 4 – MST over a *proximity* graph (Fig. 13 in paper)
// ----------------------------------------------------------------------

/**
 * Prim's variant that exactly follows the outer-loop structure of
 * Algorithm 4 lines 1-18 but with a heap for clarity.
 * 
 * Paper Reference: Algorithm 4 - MST over a proximity graph
 * 
 * @param G The proximity graph
 * @return List of MST edges as (u, v, weight) tuples
 */
std::vector<std::tuple<int, int, double>> mst_proximity(const Graph& G) {
    if (G.size() == 0) return {};
    
    // Line 1: Initialize MST
    std::vector<std::tuple<int, int, double>> mst;
    std::unordered_map<int, bool> visited;
    
    // Line 2: Choose starting node (first node in graph)
    auto nodes = G.get_nodes();
    if (nodes.empty()) return {};
    
    int start = nodes[0];
    visited[start] = true;
    
    // Line 3-6: Initialize priority queue with edges from start node
    std::priority_queue<std::tuple<double, int, int>, 
                       std::vector<std::tuple<double, int, int>>,
                       std::greater<std::tuple<double, int, int>>> edges;
    
    for (const auto& [v, w] : G.get_neighbors(start)) {
        edges.emplace(w, start, v);
    }
    
    // Line 7-18: Main MST construction loop
    while (!edges.empty() && visited.size() < G.size()) {
        auto [w, u, v] = edges.top();
        edges.pop();
        
        // Line 8-9: Skip if already visited
        if (visited.count(v) > 0) {
            continue;
        }
        
        // Line 10-12: Add edge to MST and mark as visited
        visited[v] = true;
        mst.emplace_back(u, v, w);
        
        // Line 13-18: Add new edges to priority queue
        for (const auto& [nv, nw] : G.get_neighbors(v)) {
            if (visited.count(nv) == 0) {
                edges.emplace(nw, v, nv);
            }
        }
    }
    
    return mst;  // |V|-1 edges
}

// Function to build MST from graph using Kruskal's algorithm
std::vector<std::tuple<double, int, int>> build_mst(const Graph& Gx) {
    auto nodes = Gx.get_nodes();
    std::vector<std::tuple<double, int, int>> edges;
    
    // Collect all edges from the graph
    for (int u : nodes) {
        const auto& neighbors = Gx.get_neighbors(u);
        for (const auto& [v, w] : neighbors) {
            if (u < v) {  // Avoid duplicate edges
                edges.emplace_back(w, u, v);
            }
        }
    }
    
    // Sort edges by weight
    std::sort(edges.begin(), edges.end());
    
    // DSU for cycle detection
    std::unordered_map<int, int> parent;
    std::unordered_map<int, int> rank;
    
    for (int u : nodes) {
        parent[u] = u;
        rank[u] = 0;
    }
    
    auto find = [&](int u) -> int {
        while (parent[u] != u) {
            parent[u] = parent[parent[u]];
            u = parent[u];
        }
        return u;
    };
    
    auto union_sets = [&](int u, int v) -> bool {
        int ru = find(u), rv = find(v);
        if (ru == rv) return false;
        
        if (rank[ru] < rank[rv]) {
            parent[ru] = rv;
        } else if (rank[ru] > rank[rv]) {
            parent[rv] = ru;
        } else {
            parent[rv] = ru;
            rank[ru]++;
        }
        return true;
    };
    
    // Build MST using Kruskal's algorithm
    std::vector<std::tuple<double, int, int>> mst_edges;
    for (const auto& [weight, u, v] : edges) {
        if (union_sets(u, v)) {
            mst_edges.emplace_back(weight, u, v);
        }
    }
    
    // Connectivity check: ensure MST is connected
    int num_components = 0;
    std::unordered_set<int> root_nodes;
    for (int u : nodes) {
        int root = find(u);
        root_nodes.insert(root);
    }
    num_components = root_nodes.size();
    
    // Debug: Print MST edge and node counts
    std::cout << "MST Debug: |mst_edges| = " << mst_edges.size() << ", |nodes| = " << nodes.size() << std::endl;
    
    // Assert MST properties: # nodes = # edges + 1 (tree property) and # edges = # nodes of Gx - 1
    //assert(mst_edges.size() == nodes.size() - 1 && "MST should have |V| - 1 edges");
    //assert(num_components == 1 && "MST should be connected (single component)");
    
    return mst_edges;
}

// ----------------------------------------------------------------------
// Algorithm 5 – WindowOrder (Fig. 15 in paper)
// ----------------------------------------------------------------------

/**
 * Simplified window_order that returns only the first entry for parallelism
 * 
 * @param Gx The X graph
 * @param X_vectors Dictionary of X vectors
 * @param y0 Entry point in Y
 * @param vec_y0 Vector representation of y0
 * @return Pair of (first_entry, mst_tree) where first_entry is (kappa_i, x_i) and mst_tree maps parent to children
 */
std::tuple<std::pair<int, int>, std::unordered_map<int, std::vector<int>>, std::unordered_map<int, double>, std::unordered_map<std::pair<int, int>, double>> window_order_first(
    const Graph& Gx,
    const std::unordered_map<int, std::vector<float>>& X_vectors,
    const int y0,
    const std::vector<float>& vec_y0) {

    assert(y0 == -1);  // y0 is not part of X
    
    // Line 1: sort edges (xi, y0) with ascending order of δ(xi, y0) for each xi ∈ V, and denote them as e1, e2, ..., e|X|;
    std::vector<std::pair<double, int>> synthetic_edges;  // (weight, xi) pairs
    std::unordered_map<int, double> distances_to_y0;
    
    for (const auto& [xi, xi_vec] : X_vectors) {
        if (!xi_vec.empty()) {
            double dist = l2_distance(xi_vec, vec_y0);
            distances_to_y0[xi] = dist;
            synthetic_edges.emplace_back(dist, xi);
        }
    }
    
    // Sort synthetic edges by weight (ascending order)
    std::sort(synthetic_edges.begin(), synthetic_edges.end());
    
    // Line 2: i ← 1; T ← ∅;
    int i = 0;  // Index for synthetic edges (0-based)
    std::unordered_map<int, std::vector<int>> T;  // Tree edges
    std::unordered_map<std::pair<int, int>, double> mst_edge_distances;
    
    // Line 3: assign each data point u ∈ V ∪ {y0} to the set only contains itself;
    auto nodes = Gx.get_nodes();
    std::unordered_map<int, int> parent;  // DSU parent array
    std::unordered_map<int, int> rank;    // DSU rank array
    
    for (int u : nodes) {
        parent[u] = u;
        rank[u] = 0;
    }
    parent[y0] = y0;
    rank[y0] = 0;
    
    // DSU find function
    auto find = [&](int u) -> int {
        while (parent[u] != u) {
            parent[u] = parent[parent[u]];
            u = parent[u];
        }
        return u;
    };
    
    // DSU union function
    auto union_sets = [&](int u, int v) -> bool {
        int ru = find(u), rv = find(v);
        if (ru == rv) return false;  // Would create cycle
        
        if (rank[ru] < rank[rv]) {
            parent[ru] = rv;
        } else if (rank[ru] > rank[rv]) {
            parent[rv] = ru;
        } else {
            parent[rv] = ru;
            rank[ru]++;
        }
        return true;  // Successfully merged
    };
    
    // Build MST from graph Gx
    std::vector<std::tuple<double, int, int>> mst_edges = build_mst(Gx);
    
    // Sort MST edges by weight for the algorithm
    std::sort(mst_edges.begin(), mst_edges.end());
    
    // Line 4: for each edge (p, q) in G's MST in ascending order of δ(p, q) do
    for (const auto& [edge_weight, p, q] : mst_edges) {
        
        // Line 5: while i ≤ |V| and weight of edge ei = (u, v) ≤ δ(p, q) do
        while (i < synthetic_edges.size() && synthetic_edges[i].first <= edge_weight) {
            // Line 6: if u and v are in different sets then
            int u = y0;  // synthetic edge is always (y0, xi)
            int v = synthetic_edges[i].second;
            
            if (find(u) != find(v)) {
                // Line 7: merge the sets containing u and v;
                union_sets(u, v);
                
                // Line 8: add edge (u, v) into T;
                T[u].push_back(v);
                T[v].push_back(u);
                mst_edge_distances[{u, v}] = synthetic_edges[i].first;
                mst_edge_distances[{v, u}] = synthetic_edges[i].first;
            }
            
            // Line 9: i ← i + 1;
            i++;
        }
        
        // Line 10: if p and q are in different sets then
        if (find(p) != find(q)) {
            // Line 11: merge the sets containing p and q;
            union_sets(p, q);
            
            // Line 12: add edge (p, q) into T;
            T[p].push_back(q);
            T[q].push_back(p);
            mst_edge_distances[{p, q}] = edge_weight;
            mst_edge_distances[{q, p}] = edge_weight;
        }
    }

    
    // Line 13: use DFS to derive parent node p[xi] for each xi ∈ V in T rooted at y0;
    std::unordered_map<int, int> parent_nodes;  // p[xi] for each xi
    std::unordered_set<int> visited;
    std::stack<std::pair<int, int>> dfs_stack;  // (node, parent) pairs
    
    dfs_stack.push({y0, -1});  // y0 has no parent
    
    while (!dfs_stack.empty()) {
        auto [current, current_parent] = dfs_stack.top();
        dfs_stack.pop();
        
        if (visited.count(current) > 0) continue;
        visited.insert(current);
        
        if (current != y0) {
            parent_nodes[current] = current_parent;
        }
        
        // Add children to stack
        if (T.count(current) > 0) {
            for (int child : T.at(current)) {
                if (visited.count(child) == 0) {
                    dfs_stack.push({child, current});
                }
            }
        }
    }
    
    // Line 14: return (p[xi], xi) for each xi ∈ X according to the DFS order;
    std::vector<std::pair<int, int>> window_order_list;
    for (int xi : nodes) {
        if (parent_nodes.count(xi) > 0) {
            window_order_list.emplace_back(parent_nodes[xi], xi);
        }
    }
    
    // Verify window_order by traversing through it
    std::cout << "=== Window Order Verification ===" << std::endl;
    std::cout << "Verifying window_order properties..." << std::endl;
    
    for (const auto& [parent, child] : window_order_list) {
        if (parent != y0) {
            // Check if distance(child → parent) < distance(child → y0)
            double dist_to_parent = mst_edge_distances.at({parent, child});
            double dist_to_y0 = distances_to_y0.at(child);
            
            if (dist_to_parent >= dist_to_y0) {
                std::cout << "LIMIT: parent is farther than y0 for child=" << child << std::endl;
                std::cout << "  parent=" << parent << " (not y0)" << std::endl;
                std::cout << "    distance(child → parent): " << dist_to_parent << std::endl;
                std::cout << "    distance(child → y0): " << dist_to_y0 << std::endl;
                std::cout << "    Expected: distance to parent < distance to y0" << std::endl;
                
                assert(false && "LIMIT: parent is farther than y0 for child");
            } else {
                //std::cout << "  Window order property verified successfully for child=" << child << "!" << std::endl;
            }
        } else {
            std::cout << "  Child " << child << " -> y0 (direct connection, no property check needed)" << std::endl;
        }
    }
    
    std::cout << "  ✓ All window_order properties verified successfully" << std::endl;
    
    // Find and print the parent-child pair with minimum distance
    double min_distance = std::numeric_limits<double>::max();
    std::pair<int, int> min_distance_pair = {-1, -1};
    
    for (const auto& [parent, child] : window_order_list) {
        double dist_to_parent = mst_edge_distances.at({parent, child});
        if (dist_to_parent == 0) {
            std::cout << "WARNING: Parent-child pair with distance 0: " << parent << " -> " << child << std::endl;
        }
        if (dist_to_parent < min_distance) {
            min_distance = dist_to_parent;
            min_distance_pair = {parent, child};
        }
    }
    
    std::cout << "=== Minimum Distance Parent-Child Pair ===" << std::endl;
    std::cout << "Parent: " << min_distance_pair.first << ", Child: " << min_distance_pair.second << std::endl;
    std::cout << "Minimum distance: " << min_distance << std::endl;
    
    // Find the first entry (closest to y0) for backward compatibility
    int first_x = synthetic_edges[0].second;
    
    return {{y0, first_x}, T, distances_to_y0, mst_edge_distances};
    
    // Old code
    assert(false);
    
    // Comprehensive MST verification after construction
    std::cout << "=== Comprehensive MST Verification ===" << std::endl;
    std::cout << "Verifying MST properties after construction..." << std::endl;
    
    // 1. Verify MST connectivity from root (y0)
    std::unordered_set<int> mst_nodes;
    std::queue<int> mst_bfs;
    mst_bfs.push(y0);
    mst_nodes.insert(y0);
    
    while (!mst_bfs.empty()) {
        int current = mst_bfs.front();
        mst_bfs.pop();
        
        if (T.count(current) > 0) {
            for (int child : T.at(current)) {
                if (mst_nodes.count(child) == 0) {
                    mst_nodes.insert(child);
                    mst_bfs.push(child);
                }
            }
        }
    }
    
    std::cout << "MST connectivity check:" << std::endl;
    std::cout << "  Nodes in MST: " << mst_nodes.size() << std::endl;
    std::cout << "  Expected nodes: " << (nodes.size() + 1) << " (including y0)" << std::endl;
    
    if (mst_nodes.size() != nodes.size() + 1) {
        std::cout << "ERROR: MST is not fully connected!" << std::endl;
        std::cout << "  Missing nodes: ";
        for (int node : nodes) {
            if (mst_nodes.count(node) == 0) {
                std::cout << node << " ";
            }
        }
        std::cout << std::endl;
        assert(false && "MST connectivity verification failed");
    } else {
        std::cout << "  ✓ MST is fully connected" << std::endl;
    }
    
    // 2. Verify MST property: distance(x_i → kappa) < distance(x_i → y0) if kappa ≠ y0
    std::cout << "MST property verification:" << std::endl;
    int violations = 0;
    
    for (int node : nodes) {  // Skip y0, only check X nodes
        // Find parent of this node in the MST
        int parent = -1;
        double edge_weight = 0.0;
        
        for (const auto& [p, children] : T) {
            if (std::find(children.begin(), children.end(), node) != children.end()) {
                parent = p;
                edge_weight = mst_edge_distances.at({p, node});
                break;
            }
        }
        
        if (parent != -1) {
            if (parent == y0) {
                assert(false);
                // Node is directly connected to y0 - no verification needed
                std::cout << "  Node " << node << " -> y0 (direct connection, weight: " << edge_weight << ")" << std::endl;
                std::cout << "    ✓ Direct connection to y0, no property check needed" << std::endl;
            } else {
                // Node is connected to kappa ≠ y0 - verify distance(x_i → kappa) < distance(x_i → y0)
                double direct_dist_to_y0 = distances_to_y0.at(node);
                
                //std::cout << "  Node " << node << " -> parent " << parent << " (edge weight: " << edge_weight << ")" << std::endl;
                //std::cout << "  Node " << node << " -> y0 (direct: " << direct_dist_to_y0 << ")" << std::endl;
                
                if (edge_weight >= direct_dist_to_y0) {
                    std::cout << "    ❌ MST property violation!" << std::endl;
                    std::cout << "      distance(" << node << " → " << parent << ") = " << edge_weight << std::endl;
                    std::cout << "      distance(" << node << " → y0) = " << direct_dist_to_y0 << std::endl;
                    std::cout << "      Expected: distance to parent < distance to y0" << std::endl;
                    violations++;
                } else {
                    //std::cout << "    ✓ MST property satisfied: " << edge_weight << " < " << direct_dist_to_y0 << std::endl;
                }
            }
        }
    }
    
    if (violations > 0) {
        std::cout << "ERROR: Found " << violations << " MST property violations!" << std::endl;
        assert(false && "MST property verification failed");
    } else {
        std::cout << "  ✓ All MST properties verified successfully" << std::endl;
    }
    
    std::cout << "=== End Comprehensive MST Verification ===" << std::endl;
    
    return {{y0, first_x}, T, distances_to_y0, mst_edge_distances};  // y0 as parent, first_x as child, T is the MST tree, cached distances, MST edge distances
}

// ----------------------------------------------------------------------
// Forward declarations
// ----------------------------------------------------------------------

// Apply triangle inequality pruning
// Returns: (can_prune, is_match_pruned, distance_estimate)
std::tuple<bool, bool, double> apply_triangle_inequality_pruning(
    const std::vector<float>& query_vec, int data_vector_id, double epsilon,
    const std::vector<std::vector<float>>& clusters2_centroids,
    const std::vector<std::vector<double>>& clusters2_centroid_distances,
    const std::unordered_map<int, std::pair<int, int>>& global_to_clusters2_mapping,
    const std::unordered_map<int, std::vector<float>>& global_vectors,
    std::unordered_map<int, double>& query_to_centroid_distances,
    int query_id, const std::string& call_site);

// BFS function for collecting visited nodes with depth
std::unordered_set<int> perform_bfs_with_depth_collection(
    const Graph& Gy,
    const std::vector<float>& x_i_vec,
    const std::vector<float>& x_j_vec,
    const std::unordered_set<int>& seeds,
    double epsilon,
    const std::unordered_map<int, std::vector<float>>& Y_vectors,
    int& bfs_distance_count,
    bool use_triangle_inequality,
    const std::vector<std::vector<float>>* clusters2_centroids,
    const std::vector<std::vector<double>>* clusters2_centroid_distances,
    const std::unordered_map<int, std::pair<int, int>>* global_to_clusters2_mapping,
    const std::unordered_map<int, std::vector<float>>* global_vectors,
    std::unordered_map<int, double>* query_to_centroid_distances,
    const std::vector<std::unordered_map<int, int>>* local_to_global_mapping,
    int cluster_id,
    int query_id,
    bool enable_query_to_data_edges,
    int data_offset,
    bool enable_closest_fallback,
    std::vector<std::pair<int, int>>* bfs_visited,
    std::vector<int>* final_join_results);

// ----------------------------------------------------------------------
// Algorithm 3 – JoinSlide (Fig. 10 in paper)
// ----------------------------------------------------------------------

/**
 * Exact port of lines 1-20 of Algorithm 3 - JoinSlide
 * 
 * Paper Reference: Algorithm 3 - JoinSlide
 * 
 * @param Gy The Y graph
 * @param x_i_vec Vector representation of x_i
 * @param x_j_vec Vector representation of x_j
 * @param J_i Current join window
 * @param epsilon Distance threshold
 * @param w_queue Queue size limit
 * @param Y_vectors Dictionary of Y vectors
 * @param seed_distance_count Counter for seed distance calculations
 * @param neighbor_distance_count Counter for neighbor distance calculations
 * @param bfs_distance_count Counter for BFS distance calculations
 * @param early_terminate Enable early termination when threshold is met
 * @param enable_top1_detection Enable top-1 detection for early termination
 * @param enable_closest_fallback Enable closest result fallback when J_j is empty
 *                                When enabled, if J_j is empty, returns the closest node found during processing
 * @param print_mode Enable debug output
 * @param sort_jj_by_distance Sort J_j by distance to x_j (smallest to largest) when enabled
 * @param seed_offset Filter out nodes >= seed_offset when adding to J_j (when seed_offset > 0)
 * @param closest_node_output Output parameter to store the closest node found (can be nullptr)
 * @param max_bfs_out_of_range_tolerance Max BFS out-of-range tolerance - push out-of-range points to BFS queue and terminate after N consecutive out-of-range points (0=disabled)
 * @param adaptive_bfs_threshold_factor Adaptive BFS threshold factor - dynamically adjust BFS threshold based on distance std deviation (0.0=disabled, >0.0=epsilon + factor/std_dev)
 * @return Join window J_j ⊆ Y (set of node ids)
 */
std::unordered_set<int> join_slide(
    const Graph& Gy,
    const std::vector<float>& x_i_vec,
    const std::vector<float>& x_j_vec,
    const std::unordered_set<int>& J_i,
    double epsilon,
    int w_queue,
    const std::unordered_map<int, std::vector<float>>& Y_vectors,
    int& seed_distance_count,
    int& neighbor_distance_count,
    int& bfs_distance_count,
    bool early_terminate = false,
    bool enable_top1_detection = false,
    bool enable_closest_fallback = true,
    bool print_mode = false,
    bool use_triangle_inequality = false,
    const std::vector<std::vector<float>>* clusters2_centroids = nullptr,
    const std::vector<std::vector<double>>* clusters2_centroid_distances = nullptr,
    const std::unordered_map<int, std::pair<int, int>>* global_to_clusters2_mapping = nullptr,
    const std::unordered_map<int, std::vector<float>>* global_vectors = nullptr,
    std::unordered_map<int, double>* query_to_centroid_distances = nullptr,
    const std::vector<std::unordered_map<int, int>>* local_to_global_mapping = nullptr,
    int cluster_id = -1,
    int query_id = -1, // Add query_id parameter for debugging
    bool enable_query_to_data_edges = false,
    int k_top_data_points = 0,
    int data_offset = 0,
    bool break_before_BFS = false,
    bool sort_jj_by_distance = false, // Sort J_j by distance to x_j when enabled
    int seed_offset = 0, // Filter out nodes >= seed_offset when adding to J_j
    std::unordered_set<int>* closest_node_output = nullptr, // Output parameter to store the closest node found
    std::vector<std::pair<int, int>>* greedy_visited = nullptr,
    std::vector<int>* bfs_seeds = nullptr,
    std::vector<std::pair<int, int>>* bfs_visited = nullptr,
    std::vector<int>* final_join_results = nullptr,
    int max_bfs_out_of_range_tolerance = 0,
    double adaptive_bfs_threshold_factor = 0.0,
    bool one_hop_data_only = false,
    int number_cached = 0) {

    //assert(cluster_id < 0);
    
    PROFILE_SCOPE(join_slide);

    std::set<std::pair<double, int>> closest_nodes;
    unsigned visited_count = 0;
    
    
    // Clear query-to-centroid distances cache for single-cluster mode (cluster_id == -1)
    // This prevents cross-query contamination in triangle inequality pruning
    if (cluster_id == -1 && use_triangle_inequality) {
        query_to_centroid_distances->clear();
    }
    
    if (print_mode) {
        std::cout << "=== JOIN_SLIDE DEBUG MODE ===" << std::endl;
        std::cout << "Input J_i size: " << J_i.size() << std::endl;
        std::cout << "Input J_i contents: ";
        if (J_i.empty()) {
            std::cout << "{}";
        } else {
            std::cout << "{";
            bool first = true;
            for (int y : J_i) {
                if (!first) std::cout << ", ";
                std::cout << y;
                first = false;
            }
            std::cout << "}";
        }
        std::cout << std::endl;
        std::cout << "Epsilon: " << epsilon << std::endl;
        std::cout << "W_queue: " << w_queue << std::endl;
        std::cout << "Closest result tracking: " << (enable_closest_fallback ? "ENABLED" : "DISABLED") << std::endl;
    }
    
    // Line 1: Multiset ordered by dist to x_j (closest first, farthest last)
    std::multiset<std::pair<double, int>> Q;
    
    // Initialize J_j (join window for this iteration) - store (distance, node_id) pairs
    std::vector<std::pair<double, int>> J_j_with_distances;
    std::unordered_set<int> J_j; // Keep for compatibility with existing code
    
    // Track closest result for fallback when J_j is empty
    double closest_distance = std::numeric_limits<double>::max();
    int closest_node = -1;
    bool closest_node_changed = false;

    std::unordered_map<int, bool> visited;
    
    if (seed_offset > 0) {
        int y = query_id + seed_offset;
        visited[y] = true;
        if (one_hop_data_only) {
            for (const auto& [v, _] : Gy.get_neighbors(y)) {
                visited[v] = true;
                // only add data points to the queue
                // if all dist > epsilon, it does not skip the greedy search
                if (v < seed_offset) {
                    double dist = l2_distance(Y_vectors.at(v), x_j_vec);
                    Q.insert({dist, v});
                    seed_distance_count++;
                    if (enable_closest_fallback && dist < closest_distance) {
                        closest_distance = dist;
                        closest_node = v;
                        if (print_mode) {
                            std::cout << "    1-hop new closest result: node=" << v << ", distance=" << dist << std::endl;
                        }
                    }
                    if (print_mode) {
                        std::cout << "  1-hop visited v=" << v << " (seed), distance=" << dist << std::endl;
                    }
                    if (early_terminate && dist <= epsilon) {
                        break;
                    }
                }
            }
        } else {
            Q.insert({0.0, y}); // this skips the greedy search and goes straight to the BFS
        }
    }
    // now check J_i that contains y0 or window
    {
        // find a good seed from J_i
        PROFILE_SCOPE(join_slide_line1_seed_distances);
        for (int y : J_i) {
            // Check for shutdown during distance calculations
            if (g_shutdown_requested) {
                return std::unordered_set<int>();
            }

            if (one_hop_data_only && visited[y]) {
                continue;
            }
            
            try {
                const std::vector<float>& y_vec = Y_vectors.at(y);
                if (!y_vec.empty()) {
                    double dist = 0.0;
                    visited[y] = true;
                    
                    
                    // No triangle inequality pruning, compute distance normally
                    dist = l2_distance(y_vec, x_j_vec);
                    seed_distance_count++; // Line 676 equivalent
                    
                    
                    Q.insert({dist, y});
                    
                    // Track closest result (only if it would be allowed in J_j)
                    if (enable_closest_fallback && dist < closest_distance) {                        
                        closest_distance = dist;
                        closest_node = y;
                        if (print_mode) {
                            std::cout << "    Seed new closest result: node=" << y << ", distance=" << dist << std::endl;
                        }
                    }
                    
                    if (print_mode) {
                        std::cout << "  Seed visited y=" << y << " (seed), distance=" << dist << std::endl;
                    }
                    
                    // Early termination: if any y makes dist <= threshold, break immediately
                    if (early_terminate && dist <= epsilon) {
                        break;
                    }
                }
            } catch (const std::out_of_range&) {
                // Skip if y not found
            }
        }
    }
    
    // Greedy search Lines 2-12: Handle case where no seeds within epsilon, XXX bottleneck!!!
    if (Q.empty() || Q.begin()->first > epsilon) {
        
        PROFILE_SCOPE(join_slide_lines2_12_queue_expansion);
        
        // Track minimum distance to detect when we've found top-1 nearest node
        double prev_min_distance = std::numeric_limits<double>::max();
        int iterations_without_improvement = 0;
        const int max_iterations_without_improvement = 10;
        int data_nodes_found_this_iteration = 0;  // Count data nodes found but not added to Q
        
        // Track depth for greedy search (iteration level)
        int greedy_depth = 0;
        
        while (!Q.empty()) {
            if (print_mode) {
                std::cout << "Q status: " << (Q.empty() ? "empty" : "min_dist=" + std::to_string(Q.begin()->first)) << std::endl;
            }

            auto [d_top, u] = *Q.begin();
            if (d_top <= epsilon) {
                break;
            }
            
            // Increment depth for this iteration
            greedy_depth++;
            
            if (enable_top1_detection) {
                // Check if minimum distance has improved
                if (d_top < prev_min_distance) {
                    prev_min_distance = d_top;
                    iterations_without_improvement = 0;
                    if (print_mode) {
                        std::cout << "  Top1 detection: distance improved to " << d_top << ", resetting counter" << std::endl;
                    }
                } else {
                    iterations_without_improvement++;
                    if (print_mode) {
                        std::cout << "  Top1 detection: no improvement (iterations=" << iterations_without_improvement << "/" << max_iterations_without_improvement << ")" << std::endl;
                    }
                    // If minimum distance hasn't decreased for 10 iterations, we've likely found top-1
                    if (iterations_without_improvement >= max_iterations_without_improvement) {
                        if (print_mode) {
                            std::cout << "  *** TOP1 DETECTION: Terminating due to no improvement for " << max_iterations_without_improvement << " iterations ***" << std::endl;
                        }
                        break;
                    }
                }
                
            }
            
            // Reset the counter for data nodes found this iteration
            data_nodes_found_this_iteration = 0;
            
            Q.erase(Q.begin());
            
            // Lines 6-9: Explore neighbors
            {
                PROFILE_SCOPE(join_slide_neighbor_iteration);
                if (print_mode) {
                    std::cout << "  Exploring neighbors of u=" << u << " (d_top=" << d_top << ")" << std::endl;
                }
                
                //XXX does the neighbors order matter? Or should we break if we find any neighbor closer than d_top?
                for (const auto& [v, _] : Gy.get_neighbors(u)) {
                    // Track query-to-data and query-to-query edge traversal
                    if (enable_query_to_data_edges && data_offset > 0) {
                        bool u_is_query = (u >= data_offset);
                        bool v_is_query = (v >= data_offset);
                        if (u_is_query && v_is_query) {
                            // Both endpoints are in query space
                            g_query_to_query_edges_traversed++;
                        } else if (u_is_query != v_is_query) {
                            // One is query, one is data
                            g_query_to_data_edges_traversed++;
                        } else {
                            // Both are data points
                            g_data_to_data_edges_traversed++;
                        }
                    }
                    
                    {
                        // 20% of time
                        PROFILE_SCOPE(join_slide_visited_check);
                        if (visited[v]) {
                            if (print_mode) {
                                std::cout << "    Neighbor v=" << v << " already visited, skipping" << std::endl;
                            }
                            continue;
                        }
                    }
                    
                    {
                        // 80% of time
                        PROFILE_SCOPE(join_slide_y_vectors_lookup);
                        try {
                            const std::vector<float>& vec_v = Y_vectors.at(v);
                            if (!vec_v.empty()) {
                                double dv = 0.0;
                                
                                // Apply triangle inequality pruning if enabled
                                if (use_triangle_inequality) {
                                    // Convert clusters1 local ID to global ID for triangle inequality pruning
                                    int global_v_id = v;
                                    if (cluster_id >= 0) {
                                        global_v_id = local_to_global_mapping->at(cluster_id).at(v);
                                    }
                                    
                                    auto [can_prune, is_match_pruned, dist_estimate] = apply_triangle_inequality_pruning(
                                        //x_j_vec, global_v_id, epsilon,
                                        x_j_vec, global_v_id, d_top, // here we only need to compare it with d_top, not epsilon
                                        *clusters2_centroids, *clusters2_centroid_distances,
                                        *global_to_clusters2_mapping, *global_vectors,
                                        *query_to_centroid_distances, query_id, "NEIGHBOR");
                                    
                                    if (can_prune) {
                                        dv = dist_estimate;
                                    } else {
                                        // Need to compute actual distance
                                        dv = l2_distance(vec_v, x_j_vec);
                                        neighbor_distance_count++; // Line 719 equivalent
                                    }
                                } else {
                                    // No triangle inequality pruning, compute distance normally
                                    dv = l2_distance(vec_v, x_j_vec);
                                    neighbor_distance_count++; // Line 719 equivalent
                                }
                                
                                // Track closest result (only if it would be allowed in J_j)
                                if (enable_closest_fallback && dv < closest_distance) {
                                    if (seed_offset == 0 || v < seed_offset) {
                                        closest_distance = dv;
                                        closest_node = v;
                                        if (print_mode) {
                                            std::cout << "    Greedy new closest result: node=" << v << ", distance=" << dv << std::endl;
                                        }
                                    }
                                }
                                
                                if (print_mode) {
                                    std::cout << "    Neighbor v=" << v << ", distance=" << dv;
                                }
                                
                                if (dv < d_top) { //XXX changed from < to <= (to see if it fixes the issue for NYTimes)
                                    Q.insert({dv, v});
                                    if (print_mode) {
                                        std::cout << " -> added to Q" << std::endl;
                                    }
                                    //visited[v] = true;
                                } else {
                                    // Node was computed but not added to Q - this is a data node we found
                                    data_nodes_found_this_iteration++;
                                    if (print_mode) {
                                        std::cout << " -> not added (dv >= d_top)" << std::endl;
                                    }
                                }
                                visited[v] = true; //XXX moved to here, algorithm in the paper is wrong!!!
                                
                                // Collect visited nodes during greedy search if enabled
                                if (greedy_visited != nullptr) {
                                    greedy_visited->push_back({v, greedy_depth});
                                }
                            }
                        } catch (const std::out_of_range&) {
                            if (print_mode) {
                                std::cout << "    Neighbor v=" << v << " vector not found, skipping" << std::endl;
                            }
                            // Vector not found, skip
                        }
                    }
                }
            }
            
            // Lines 10-11: Cap the queue - remove farthest vectors
            while (Q.size() > w_queue) { //XXX if to while
                // Remove the farthest vector (last element in multiset)
                auto it = Q.end();
                --it;
                Q.erase(it);
            }
        }
        
        if (Q.empty()) {
            // If Q is empty but we found a closest result, return it
            if (enable_closest_fallback && closest_node != -1) {
                assert(seed_offset == 0 || closest_node < seed_offset);
                std::unordered_set<int> fallback_result;
                fallback_result.insert(closest_node);
                if (print_mode) {
                    std::cout << "Q is empty, returning closest result: node=" << closest_node << ", distance=" << closest_distance << std::endl;
                }
                return fallback_result;
            }
            return std::unordered_set<int>();  // Line 12
        }
    }
    else {
        if (print_mode) {
            std::cout << "Q status: " << (Q.empty() ? "empty" : "min_dist=" + std::to_string(Q.begin()->first)) << std::endl;
        }
    }

    // Line 13: Initialize J_j (already declared above)
    {
        
        PROFILE_SCOPE(join_slide_line13_collect_seeds);
        if (print_mode) {
            std::cout << "Initializing J_j from Q (size=" << Q.size() << ")" << std::endl;
        }
        
        while (!Q.empty()) {
            auto [d, y] = *Q.begin();
            if (d <= epsilon) {                
                J_j.insert(y);
                J_j_with_distances.emplace_back(d, y); // Store distance with node
                if (print_mode) {
                    std::cout << "  J_j: added y=" << y << " (distance=" << d << ")" << std::endl;
                }
            } else {
                if (print_mode) {
                    std::cout << "  J_j: skipped y=" << y << " (distance=" << d << " > epsilon=" << epsilon << ")" << std::endl;
                }
            }
            Q.erase(Q.begin());
        }
        
        if (print_mode) {
            std::cout << "J_j final size: " << J_j.size() << std::endl;
        }
    }

    // Collect BFS seeds (J_j at this point)
    if (bfs_seeds != nullptr) {
        *bfs_seeds = std::vector<int>(J_j.begin(), J_j.end());
    }
    
    // Early termination for BFS data collection
    if (break_before_BFS) {
        // Ensure closest node is stored in output parameter
        if (closest_node_output != nullptr && closest_node != -1) {
            closest_nodes.insert({closest_distance, closest_node});
        }
        
        // Return J_j as the seeds, BFS will be done separately
        if (sort_jj_by_distance) {
            // Return sorted J_j by distance to x_j (smallest to largest)
            std::sort(J_j_with_distances.begin(), J_j_with_distances.end());
            
            std::unordered_set<int> sorted_jj;
            for (const auto& [dist, y] : J_j_with_distances) {
                sorted_jj.insert(y);
            }
            return sorted_jj;
        } else {
            return J_j;
        }
    }
    
    // Lines 14-19: BFS expansion
    {
        PROFILE_SCOPE(join_slide_lines14_19_bfs_expansion);
        std::queue<std::pair<int, int>> bfs;  // (node, depth)
        for (int seed : J_j) {
            bfs.push({seed, 0});
            // Add initial seeds to bfs_visited if collection is enabled
            if (bfs_visited != nullptr) {
                bfs_visited->push_back({seed, 0});
            }
        }
        
        // BFS out-of-range tolerance counter
        int bfs_out_of_range_count = 0;
        
        // Adaptive threshold variables
        std::vector<double> distances_for_std;
        double adaptive_threshold = epsilon;
        bool adaptive_threshold_computed = false;
        
        // Helper function to compute adaptive threshold
        auto compute_adaptive_threshold = [&]() -> double {
            if (print_mode) {
                std::cout << "  Adaptive threshold: " << epsilon << " + " << adaptive_bfs_threshold_factor << std::endl;
            }
            return epsilon + adaptive_bfs_threshold_factor; // DEBUG

            if (distances_for_std.size() < 2) return epsilon;
            
            // Use epsilon as the mean instead of computing from data
            double mean = epsilon;
            
            // Compute standard deviation using filtered distances and epsilon as mean
            double variance = 0.0;
            for (double d : distances_for_std) {
                variance += (d - mean) * (d - mean);
            }
            double std_dev = std::sqrt(variance / distances_for_std.size());
            
            // Compute adaptive threshold: epsilon + (factor / std_dev)
            double adaptive_thresh = epsilon + (adaptive_bfs_threshold_factor / std_dev);
            
            if (print_mode) {
                std::cout << "  Adaptive threshold: distances_for_std=" << distances_for_std.size() 
                         << ", mean=epsilon=" << mean << ", std=" << std_dev 
                         << ", factor=" << adaptive_bfs_threshold_factor 
                         << ", adaptive_thresh=" << adaptive_thresh << std::endl;
            }
            
            return adaptive_thresh;
        };
        
        if (print_mode) {
            std::cout << "Starting BFS expansion with " << bfs.size() << " seeds" << std::endl;
        }
        
        while (!bfs.empty() && !g_shutdown_requested) {
            auto [yp, depth] = bfs.front();
            bfs.pop();
            
            if (print_mode) {
                std::cout << "  BFS: processing yp=" << yp << std::endl;
            }
            
            for (const auto& [yq, _] : Gy.get_neighbors(yp)) {
                // Track query-to-data and query-to-query edge traversal
                if (enable_query_to_data_edges && data_offset > 0) {
                    bool yp_is_query = (yp >= data_offset);
                    bool yq_is_query = (yq >= data_offset);
                    if (yp_is_query && yq_is_query) {
                        // Both endpoints are in query space
                        g_query_to_query_edges_traversed++;
                    } else if (yp_is_query != yq_is_query) {
                        // One is query, one is data
                        g_query_to_data_edges_traversed++;
                    } else {
                        // Both are data points
                        g_data_to_data_edges_traversed++;
                    }
                }
                
                // Check for shutdown during BFS expansion
                if (g_shutdown_requested) {
                    break;
                }

                // Don't need to follow!!!
                if (seed_offset > 0 && yq >= seed_offset) {
                    if (print_mode) {
                        std::cout << "    BFS neighbor yq=" << yq;
                        std::cout << " -> not added (yq >= seed_offset=" << seed_offset << ")" << std::endl;
                    }
                    continue;
                }

                //XXX check visited[yq], this is not in the paper!!!
                // it's safe, since the visited nodes in the greedy phase are farther than epsilon
                if (visited[yq]) {
                    if (print_mode) {
                        std::cout << "    BFS neighbor yq=" << yq << " already visited, skipping" << std::endl;
                    }
                    continue;
                }

                visited[yq] = true;

                if (bfs_visited != nullptr) {
                    bfs_visited->push_back({yq, depth + 1});
                }
                
                if (J_j.count(yq) == 0) {
                    try {
                        const std::vector<float>& yq_vec = Y_vectors.at(yq);
                        if (!yq_vec.empty()) {
                            double dist = 0.0;
                            bool is_match = false;
                            
                            // Apply triangle inequality pruning if enabled
                            if (use_triangle_inequality) {
                                // Convert clusters1 local ID to global ID for triangle inequality pruning
                                int global_yq_id = yq;
                                if (cluster_id >= 0) {
                                    global_yq_id = local_to_global_mapping->at(cluster_id).at(yq);
                                }
                                
                                auto [can_prune, is_match_pruned, dist_estimate] = apply_triangle_inequality_pruning(
                                    x_j_vec, global_yq_id, epsilon,
                                    *clusters2_centroids, *clusters2_centroid_distances,
                                    *global_to_clusters2_mapping, *global_vectors,
                                    *query_to_centroid_distances, query_id, "BFS");
                                
                                if (can_prune) {
                                    dist = dist_estimate;
                                    is_match = is_match_pruned;
                                } else {
                                    // Need to compute actual distance
                                    dist = l2_distance(yq_vec, x_j_vec);
                                    bfs_distance_count++; // Line 782 equivalent
                                    is_match = (dist <= epsilon);
                                }
                            } else {
                                // No triangle inequality pruning, compute distance normally
                                dist = l2_distance(yq_vec, x_j_vec);
                                bfs_distance_count++; // Line 782 equivalent
                                is_match = (dist <= epsilon);
                            }
                            
                            // Track closest result (only if it would be allowed in J_j)
                            if (enable_closest_fallback && dist < closest_distance) {
                                // Only consider nodes that would pass seed_offset filtering
                                if (seed_offset == 0 || yq < seed_offset) {
                                    closest_distance = dist;
                                    closest_node = yq;
                                    closest_node_changed = true;
                                    if (print_mode) {
                                        std::cout << "    BFS new closest result: node=" << yq << ", distance=" << dist << std::endl;
                                    }
                                }
                                else {
                                    assert(false);
                                }
                            }
                            
                            // Collect distances for adaptive threshold computation
                            if (adaptive_bfs_threshold_factor > 0.0) {
                                //if (std::abs(dist - epsilon) <= 0.1 * epsilon) {
                                if (true) {
                                    distances_for_std.push_back(dist);
                                    
                                    // Compute adaptive threshold after collecting some distances
                                    if (distances_for_std.size() >= 10 && !adaptive_threshold_computed) {
                                        adaptive_threshold = compute_adaptive_threshold();
                                        adaptive_threshold_computed = true;
                                        if (print_mode) {
                                            std::cout << "  Adaptive threshold computed: " << adaptive_threshold << " (epsilon=" << epsilon << ")" << std::endl;
                                        }
                                    }
                                }
                            }
                            
                            if (print_mode) {
                                std::cout << "    BFS neighbor yq=" << yq << ", distance=" << dist;
                            }
                            
                            if (is_match) {
                                // Filter out nodes >= seed_offset when seed_offset > 0
                                if (seed_offset == 0 || yq < seed_offset) {
                                    J_j.insert(yq);
                                    J_j_with_distances.emplace_back(dist, yq); // Store distance with node
                                    bfs.push({yq, depth + 1});
                                    if (print_mode) {
                                        std::cout << " -> added to J_j and BFS queue" << std::endl;
                                    }

                                } else {
                                    assert(false);
                                    if (print_mode) {
                                        std::cout << " -> not added (yq >= seed_offset=" << seed_offset << ")" << std::endl;
                                    }
                                }
                            } else {
                                // Check if we should push to BFS queue based on adaptive threshold or tolerance
                                bool should_push_to_bfs = false;
                                
                                // Check adaptive threshold first
                                if (adaptive_bfs_threshold_factor > 0.0 && dist <= adaptive_threshold) {
                                    should_push_to_bfs = true;
                                    if (print_mode) {
                                        std::cout << " -> out-of-range but within adaptive threshold (" << dist << " <= " << adaptive_threshold << "), pushed to BFS queue" << std::endl;
                                    }
                                }
                                // Check tolerance-based pushing
                                else if (max_bfs_out_of_range_tolerance > 0 && bfs_out_of_range_count < max_bfs_out_of_range_tolerance) {
                                    should_push_to_bfs = true;
                                    bfs_out_of_range_count++;
                                    if (print_mode) {
                                        std::cout << " -> out-of-range, pushed to BFS queue (count=" << bfs_out_of_range_count << ")" << std::endl;
                                    }
                                }
                                
                                if (should_push_to_bfs) {
                                    bfs.push({yq, depth + 1});
                                } else {
                                    if (print_mode) {
                                        std::cout << " -> out-of-range, not pushed (count=" << bfs_out_of_range_count << ", max=" << max_bfs_out_of_range_tolerance << ", adaptive_thresh=" << adaptive_threshold << ")" << std::endl;
                                    }
                                }
                            }

                            // Keep number_cached closest nodes cached
                            if (closest_node_output != nullptr && number_cached > 0) {
                                if (closest_nodes.size() <= number_cached) {
                                    closest_nodes.insert({dist, yq});
                                } else {
                                    if (dist < closest_nodes.rbegin()->first) {
                                        closest_nodes.erase(std::prev(closest_nodes.end()));
                                        closest_nodes.insert({dist, yq});
                                    }
                                }
                            }

                        }
                    } catch (const std::out_of_range&) {
                        if (print_mode) {
                            std::cout << "    BFS neighbor yq=" << yq << " vector not found, skipping" << std::endl;
                        }
                        // Skip if yq not found
                    }
                } else {
                    if (print_mode) {
                        std::cout << "    BFS neighbor yq=" << yq << " already in J_j, skipping" << std::endl;
                    }
                }
            }
        }
        
        if (print_mode) {
            std::cout << "BFS expansion complete. Final J_j size: " << J_j.size() << std::endl;
            std::cout << "=== DISTANCE COMPUTATION SUMMARY ===" << std::endl;
            std::cout << "Seed distances: " << seed_distance_count << std::endl;
            std::cout << "Neighbor distances: " << neighbor_distance_count << std::endl;
            std::cout << "BFS distances: " << bfs_distance_count << std::endl;
            std::cout << "Total distances: " << (seed_distance_count + neighbor_distance_count + bfs_distance_count) << std::endl;
        }
        
        // If J_j is empty and we found a closest result, add it as fallback
        if (enable_closest_fallback && J_j.empty() && closest_node != -1) {
            // Filter out nodes >= seed_offset when seed_offset > 0
            assert(seed_offset == 0 || closest_node < seed_offset);
            J_j.insert(closest_node);
            J_j_with_distances.emplace_back(closest_distance, closest_node); // Store distance with node
            if (print_mode) {
                std::cout << "J_j was empty, adding closest result: node=" << closest_node << ", distance=" << closest_distance << std::endl;
            }
        }
        
        // Collect final join results (J_j after pruning with epsilon) if enabled
        if (final_join_results != nullptr) {
            *final_join_results = std::vector<int>(J_j.begin(), J_j.end());
        }
        
        unsigned cached_size = 0;
        for (const auto& p : closest_nodes) {
            closest_node_output->insert(p.second);
            cached_size++;
        }

        std::cout << "closest_nodes.size()=" << cached_size << std::endl;
        
        if (sort_jj_by_distance) {
            // Return sorted J_j by distance to x_j (smallest to largest)
            std::sort(J_j_with_distances.begin(), J_j_with_distances.end());
            
            std::unordered_set<int> sorted_jj;
            for (const auto& [dist, y] : J_j_with_distances) {
                sorted_jj.insert(y);
            }
            return sorted_jj;
        } else {
            return J_j;
        }
    }  // Line 20
}

// BFS function for collecting visited nodes with depth
// @param max_bfs_out_of_range_tolerance Max BFS out-of-range tolerance - push out-of-range points to BFS queue and terminate after N consecutive out-of-range points (0=disabled)
// @param adaptive_bfs_threshold_factor Adaptive BFS threshold factor - dynamically adjust BFS threshold based on distance std deviation (0.0=disabled, >0.0=epsilon + factor/std_dev)
std::unordered_set<int> perform_bfs_with_depth_collection(
    const Graph& Gy,
    const std::vector<float>& x_i_vec,
    const std::vector<float>& x_j_vec,
    const std::unordered_set<int>& seeds,
    double epsilon,
    const std::unordered_map<int, std::vector<float>>& Y_vectors,
    int& bfs_distance_count,
    bool use_triangle_inequality,
    const std::vector<std::vector<float>>* clusters2_centroids,
    const std::vector<std::vector<double>>* clusters2_centroid_distances,
    const std::unordered_map<int, std::pair<int, int>>* global_to_clusters2_mapping,
    const std::unordered_map<int, std::vector<float>>* global_vectors,
    std::unordered_map<int, double>* query_to_centroid_distances,
    const std::vector<std::unordered_map<int, int>>* local_to_global_mapping,
    int cluster_id,
    int query_id,
    bool enable_query_to_data_edges,
    int data_offset,
    bool enable_closest_fallback,
    int seed_offset = 0, // Filter out nodes >= seed_offset when adding to J_j
    bool sort_jj_by_distance = false, // Sort J_j by distance to x_j when enabled
    int* closest_node_output = nullptr, // Output parameter to store the closest node found
    std::vector<std::pair<int, int>>* bfs_visited = nullptr,
    std::vector<int>* final_join_results = nullptr,
    int max_bfs_out_of_range_tolerance = 0,
    double adaptive_bfs_threshold_factor = 0.0) {
    
    std::unordered_set<int> J_j = seeds;  // Start with seeds
    std::vector<std::pair<double, int>> J_j_with_distances;
    std::unordered_set<int> visited;
    std::queue<std::pair<int, int>> bfs_queue;  // (node, depth)
    
    // Closest fallback tracking
    double closest_distance = std::numeric_limits<double>::max();
    int closest_node = -1;
    
    // Initialize BFS queue with seeds at depth 0
    for (int seed : seeds) {
        bfs_queue.push({seed, 0});
        visited.insert(seed);
        if (bfs_visited != nullptr) {
            bfs_visited->push_back({seed, 0});
        }
    }
    
    // BFS out-of-range tolerance counter
    int bfs_out_of_range_count = 0;
    
    // Adaptive threshold variables
    std::vector<double> distances_for_std;
    double adaptive_threshold = epsilon;
    bool adaptive_threshold_computed = false;
    
    // Helper function to compute adaptive threshold
    auto compute_adaptive_threshold = [&]() -> double {
        if (distances_for_std.size() < 2) return epsilon;
        
        // Use epsilon as the mean instead of computing from data
        double mean = epsilon;
        
        // Compute standard deviation using filtered distances and epsilon as mean
        double variance = 0.0;
        for (double d : distances_for_std) {
            variance += (d - mean) * (d - mean);
        }
        double std_dev = std::sqrt(variance / distances_for_std.size());
        
        // Compute adaptive threshold: epsilon + (factor / std_dev)
        double adaptive_thresh = epsilon + (adaptive_bfs_threshold_factor / std_dev);
        
        return adaptive_thresh;
    };
    
    // BFS expansion - exactly matching original logic
    while (!bfs_queue.empty()) {
        auto [yp, depth] = bfs_queue.front();
        bfs_queue.pop();
        
        const auto& neighbors = Gy.get_neighbors(yp);
        for (const auto& [yq, _] : neighbors) {
            // Track query-to-data and query-to-query edge traversal (same as original)
            if (enable_query_to_data_edges && data_offset > 0) {
                bool yp_is_query = (yp >= data_offset);
                bool yq_is_query = (yq >= data_offset);
                if (yp_is_query && yq_is_query) {
                    // Both endpoints are in query space
                    g_query_to_query_edges_traversed++;
                } else if (yp_is_query != yq_is_query) {
                    // One is query, one is data
                    g_query_to_data_edges_traversed++;
                } else {
                    // Both are data points
                    g_data_to_data_edges_traversed++;
                }
            }
            
            // Check if already visited (same as original)
            if (visited.count(yq)) {
                continue;  // Already visited
            }
            
            visited.insert(yq);
            
            // Only process if not already in J_j (same as original)
            if (J_j.count(yq) == 0) {
                try {
                    const std::vector<float>& yq_vec = Y_vectors.at(yq);
                    if (!yq_vec.empty()) {
                        double dist = 0.0;
                        bool is_match = false;
                        
                        // Apply triangle inequality pruning if enabled (same as original)
                        if (use_triangle_inequality) {
                            // Convert clusters1 local ID to global ID for triangle inequality pruning
                            int global_yq_id = yq;
                            if (cluster_id >= 0) {
                                global_yq_id = local_to_global_mapping->at(cluster_id).at(yq);
                            }
                            
                            auto [can_prune, is_match_pruned, dist_estimate] = apply_triangle_inequality_pruning(
                                x_j_vec, global_yq_id, epsilon,
                                *clusters2_centroids, *clusters2_centroid_distances,
                                *global_to_clusters2_mapping, *global_vectors,
                                *query_to_centroid_distances, query_id, "BFS");
                            
                            if (can_prune) {
                                dist = dist_estimate;
                                is_match = is_match_pruned;
                            } else {
                                // Need to compute actual distance (same as original)
                                dist = l2_distance(yq_vec, x_j_vec);
                                bfs_distance_count++;
                                is_match = (dist <= epsilon);
                            }
                        } else {
                            // No triangle inequality pruning, compute distance normally (same as original)
                            dist = l2_distance(yq_vec, x_j_vec);
                            bfs_distance_count++;
                            is_match = (dist <= epsilon);
                        }
                        
                        // Track closest result (only if it would be allowed in J_j)
                        if (enable_closest_fallback && dist < closest_distance) {
                            // Only consider nodes that would pass seed_offset filtering
                            if (seed_offset == 0 || yq < seed_offset) {
                                closest_distance = dist;
                                closest_node = yq;
                            }
                        }
                        
                        // Collect distances for adaptive threshold computation
                        if (adaptive_bfs_threshold_factor > 0.0) {
                            if (std::abs(dist - epsilon) <= 0.1 * epsilon) {
                                distances_for_std.push_back(dist);
                                
                                // Compute adaptive threshold after collecting some distances
                                if (distances_for_std.size() >= 10 && !adaptive_threshold_computed) {
                                    adaptive_threshold = compute_adaptive_threshold();
                                    adaptive_threshold_computed = true;
                                }
                            }
                        }
                        
                        if (is_match) {
                            // Filter out nodes >= seed_offset when seed_offset > 0
                            if (seed_offset == 0 || yq < seed_offset) {
                                J_j.insert(yq);
                                J_j_with_distances.emplace_back(dist, yq); // Store distance with node
                                // Add to BFS queue for further expansion
                                bfs_queue.push({yq, depth + 1});
                                if (bfs_visited != nullptr) {
                                    bfs_visited->push_back({yq, depth + 1});
                                }
                            }
                        } else {
                            // Check if we should push to BFS queue based on adaptive threshold or tolerance
                            bool should_push_to_bfs = false;
                            
                            // Check adaptive threshold first
                            if (adaptive_bfs_threshold_factor > 0.0 && dist <= adaptive_threshold) {
                                should_push_to_bfs = true;
                            }
                            // Check tolerance-based pushing
                            else if (max_bfs_out_of_range_tolerance > 0 && bfs_out_of_range_count < max_bfs_out_of_range_tolerance) {
                                should_push_to_bfs = true;
                                bfs_out_of_range_count++;
                            }
                            
                            if (should_push_to_bfs) {
                                bfs_queue.push({yq, depth + 1});
                                if (bfs_visited != nullptr) {
                                    bfs_visited->push_back({yq, depth + 1});
                                }
                            }
                        }
                    }
                } catch (const std::out_of_range&) {
                    // Skip if yq not found
                }
            }
        }
    }
    
    // Closest fallback logic (same as original)
    if (enable_closest_fallback && J_j.empty() && closest_node != -1) {
        // Filter out nodes >= seed_offset when seed_offset > 0
        assert(seed_offset == 0 || closest_node < seed_offset);
        J_j.insert(closest_node);
        J_j_with_distances.emplace_back(closest_distance, closest_node); // Store distance with node
    }
    
    // Collect final join results (J_j after pruning with epsilon) if enabled
    if (final_join_results != nullptr) {
        *final_join_results = std::vector<int>(J_j.begin(), J_j.end());
    }
    
    // Ensure closest node is stored in output parameter
    if (closest_node_output != nullptr && closest_node != -1) {
        *closest_node_output = closest_node;
    }
    
    if (sort_jj_by_distance) {
        // Return sorted J_j by distance to x_j (smallest to largest)
        std::sort(J_j_with_distances.begin(), J_j_with_distances.end());
        
        std::unordered_set<int> sorted_jj;
        for (const auto& [dist, y] : J_j_with_distances) {
            sorted_jj.insert(y);
        }
        return sorted_jj;
    } else {
        return J_j;
    }
}

std::unordered_set<int> greedy_top_k_search(
    Graph& Gy,
    int start_node, 
    const std::vector<float>& query_vec, 
    const std::unordered_map<int, std::vector<float>>& Y_vectors,
    int L,
    int max_stall_iterations,
    double epsilon,
    int seed_offset,
    int& distance_computations) {
    // ---------------------------------------------------------
    // 1. Initialization
    // ---------------------------------------------------------
    
    PROFILE_SCOPE(topK);

    // Queue
    std::multiset<std::pair<double, int>> Q;
    
    std::vector<int> result_indices;

    std::unordered_map<int, bool> visited;

    start_node = start_node + seed_offset;

    // Add start node
    double start_dist = l2_distance(Y_vectors.at(start_node), query_vec);
    // std::cout << "Start node: " << start_node << " L: " << L << " max_stall_iterations: " << max_stall_iterations << " epsilon: " << epsilon << " seed_offset: " << seed_offset << std::endl;    
    if (start_dist != 0.0) {
        std::cout << "Start dist not zero!" << start_dist << std::endl;
        std::flush(std::cout);
        assert(false);
    }
    //assert(start_dist == 0.0);
    //std::cout << "Start dist not zero!" << start_dist << std::endl;

    //Q.insert({start_dist, start_node});
    visited[start_node] = true;
    
    for (const auto& [v, _] : Gy.get_neighbors(start_node)) {
        if (v >= seed_offset) continue;

        try {
            const std::vector<float>& vec_v = Y_vectors.at(v);
            double d_v = l2_distance(vec_v, query_vec);
            distance_computations++;
            visited[v] = true;

            // add in-range points unlimitedly
            if (d_v <= epsilon) {
                Q.insert({d_v, v});
                result_indices.push_back(v);
            }
            else if (Q.size() < L) {
                Q.insert({d_v, v});
            } 
            else if (d_v < Q.rbegin()->first) { // if L = 0, then this results an error, since Q.rbegin()->first is invalid
                Q.erase(std::prev(Q.end())); // remove last element from Q, so it can never be longer than L
                Q.insert({d_v, v});           
            }
        } catch (...) { continue; }
    }

    // no in-range point found in the neighborhood of the start node, early terminate!
    if (result_indices.size() == 0)
        return std::unordered_set<int>();
    
    double prev_farthest_dist = std::numeric_limits<double>::max();
    int prev_queue_size = 0;
    int stall_count = 0;

    // ---------------------------------------------------------
    // 2. Search Loop
    // ---------------------------------------------------------
    while (!Q.empty()) {
        double current_farthest_dist = Q.rbegin()->first;
        int current_queue_size = Q.size();
        bool reset_stall_count = false;

        // Terminate if max distance in queue hasn't improved 
        if (current_farthest_dist < prev_farthest_dist) {
            reset_stall_count = true;
        }
        // And queue size hasn't increased
        if (current_queue_size > prev_queue_size) { 
            reset_stall_count = true;
        }
        if (reset_stall_count) {
            stall_count = 0;
        } else {
            stall_count++;
        }

        prev_farthest_dist = current_farthest_dist;
        prev_queue_size = current_queue_size;

        if (stall_count >= max_stall_iterations) {
            // Terminate if min distance in queue exceeds epsilon
            double min_dist = Q.begin()->first;
            if (min_dist > epsilon)
                break;
        }
        
        auto it = Q.begin();
        int u = it->second;
        Q.erase(it);

        // Expand Neighbors 
        for (const auto& [v, _] : Gy.get_neighbors(u)) {

            if (v >= seed_offset) continue; // XXX: enable SOF, this improved recall (removing unnecessary queries from the queue) and efficiency
            
            if (visited[v]) continue;

            try {
                const std::vector<float>& vec_v = Y_vectors.at(v);
                double d_v = l2_distance(vec_v, query_vec);
                distance_computations++;
                visited[v] = true;

                // add in-range points unlimitedly
                if (d_v <= epsilon) {
                    Q.insert({d_v, v});
                    result_indices.push_back(v);
                }
                else if (Q.size() < L) {
                    Q.insert({d_v, v});
                } 
                else if (d_v < Q.rbegin()->first) { // if L = 0, then this results an error, since Q.rbegin()->first is invalid
                    Q.erase(std::prev(Q.end())); // remove last element from Q, so it can never be longer than L
                    Q.insert({d_v, v});           
                }
            } catch (...) { continue; }
        }
    }

    // ---------------------------------------------------------
    // 3. Flush after termination --> we don't have any in-range point in Q left
    // ---------------------------------------------------------
    /*for (const auto& pair : Q) {
        int node_id = pair.second;
        if (pair.first <= epsilon && node_id < seed_offset) {
            result_indices.push_back(node_id);
        }
    }*/
    
    return std::unordered_set<int>(result_indices.begin(), result_indices.end());
}


// ----------------------------------------------------------------------
// Algorithm 2 – SimJoin (Fig. 8 in paper)
// ----------------------------------------------------------------------

/**
 * High-level driver exactly following Algorithm 2 lines 1-10
 * 
 * Paper Reference: Algorithm 2 - SimJoin
 * 
 * @param X_vectors Dictionary of X vectors
 * @param Y_vectors Dictionary of Y vectors
 * @param Gx The X graph
 * @param Gy The Y graph
 * @param epsilon Distance threshold
 * @param w_queue Queue size limit
 * @param num_threads Number of threads for parallel processing
 * @param mst_filename Filename for MST data output
 * @param debug_mode Enable debug mode
 * @param join_output_filename Filename for join output
 * @param result_dir Directory for result files
 * @param force_kappa_y0 Force kappa to be y0
 * @param early_terminate Enable early termination
 * @param enable_top1_detection Enable top-1 detection
 * @param enable_closest_fallback Enable closest result fallback in join_slide
 * @param cluster_id Cluster ID for multi-cluster mode
 * @param enable_seed_offset_filtering Enable filtering of nodes >= seed_offset in J_j
 * @param sort_jj_by_distance Sort J_j by distance to x_j when enabled
 * @param cache_closest_only Cache closest node only when enable_closest_fallback is enabled
 * @param one_hop_data_only Enable data seed fallback mode
 * @return Set of matched (xi, yj) pairs
 */
void sim_join(
    const std::unordered_map<int, std::vector<float>>& X_vectors,
    std::unordered_map<int, std::vector<float>>& Y_vectors,
    const Graph& Gx,
    Graph& Gy,
    double epsilon,
    int w_queue = 64,
    int num_threads = 32,
    const std::string& mst_filename = "mst_data.txt",
    bool debug_mode = false,
    const std::string& join_output_filename = "",
    const std::string& result_dir = "",
    bool force_kappa_y0 = false,
    bool early_terminate = false,
    bool enable_top1_detection = false,
    bool enable_closest_fallback = false,
    int cluster_id = -1,
    bool use_triangle_inequality = false,
    const std::vector<std::vector<float>>* clusters2_centroids = nullptr,
    const std::vector<std::vector<double>>* clusters2_centroid_distances = nullptr,
    const std::unordered_map<int, std::pair<int, int>>* global_to_clusters2_mapping = nullptr,
    const std::unordered_map<int, std::vector<float>>* global_vectors = nullptr,
    std::unordered_map<int, double>* query_to_centroid_distances = nullptr,
    const std::vector<std::unordered_map<int, int>>* local_to_global_mapping = nullptr,
    std::unordered_map<int, std::unordered_map<int, double>>* query_to_centroid_distances_map = nullptr,
    bool enable_query_to_data_edges = false,
    int k_top_data_points = 0,
    int data_offset = 0,
    bool collect_bfs_data = false,
    bool break_before_bfs = false,
    int seed_offset = 0,
    bool enable_seed_offset_filtering = false,
    bool sort_jj_by_distance = false,
    bool cache_closest_only = false,
    int nsg_entry_point = -1,
    int max_bfs_out_of_range_tolerance = 0,
    double adaptive_bfs_threshold_factor = 0.0,
    bool one_hop_data_only = false,
    int topK = 0,
    int patience = 0,
    std::unordered_map<int, bool> ood_flags = {},
    int number_cached = 0,
    bool merged_index = false
    // bool no_gx = false
    ) {
    // const std::unordered_map<int, std::vector<int>>* id_mapping = nullptr) {
    
    PROFILE_SCOPE(sim_join);
    
    // Helper function to get the correct query_to_centroid_distances cache
    auto get_query_cache = [&](int query_id) -> std::unordered_map<int, double>* {
        if (query_to_centroid_distances_map != nullptr) {
            // Multi-cluster mode: use per-query cache
            return &(*query_to_centroid_distances_map)[query_id];
        } else {
            // Single-cluster mode: use shared cache
            return query_to_centroid_distances;
        }
    };
    
    std::cout << "Running queue-based parallel SimJoin with " << num_threads << " threads..." << std::endl;
    
    
    // Use NSG entry point if provided, otherwise fall back to heuristic
    int y0_from_Y = -1;

    {
    PROFILE_SCOPE(select_entry_point);

    if (nsg_entry_point != -1) {
        // Use the stored NSG entry point
        y0_from_Y = nsg_entry_point;
        std::cout << "Using NSG entry point: " << y0_from_Y << std::endl;
    } else {
        assert(false);
        // Fall back to heuristic: find node with highest degree
        int max_degree = -1;
        for (int node : Gy.get_nodes()) {
            int degree = Gy.get_neighbors(node).size();
            if (degree > max_degree) {
                max_degree = degree;
                y0_from_Y = node;
            }
        }
        std::cout << "Using heuristic entry point (highest degree): " << y0_from_Y << " with degree " << max_degree << std::endl;
    }
    }
    
    // Function to get seed node for a given query index
    auto get_seed_node = [&](int query_index) -> int {
        if (seed_offset > 0) {
            return query_index + seed_offset;
        }
        // Default behavior: use y0_from_Y (highest degree node)
        return y0_from_Y;
    };
    
    if (y0_from_Y == -1) return;
    
    // Validate y0_from_Y exists in Y_vectors
    try {
        const std::vector<float>& vec = Y_vectors.at(y0_from_Y);
        if (vec.empty()) {
            // Find a non-empty vector
            for (const auto& [y, vec] : Y_vectors) {
                if (!vec.empty()) {
                    y0_from_Y = y;
                    break;
                }
            }
        }
    } catch (const std::out_of_range&) {
        // Find a non-empty vector
        for (const auto& [y, vec] : Y_vectors) {
            if (!vec.empty()) {
                y0_from_Y = y;
                break;
            }
        }
    }
    
    if (y0_from_Y == -1) return;
    
    // y0 = -1 indicates it's not in X, but we get vec_y0 from Y
    int y0 = -1;  // Special marker indicating y0 is not in X
    const std::vector<float>& vec_y0 = Y_vectors.at(y0_from_Y);
    
    // Variables for MST results (only used when seed_offset == 0)
    std::pair<int, int> first_entry = {-1, -1};
    std::unordered_map<int, std::vector<int>> mst_tree;
    std::unordered_map<int, double> cached_distances_to_y0;
    std::unordered_map<std::pair<int, int>, double> mst_edge_distances;
    
    // Window order duration (for timing summary)
    std::chrono::microseconds window_order_duration(0);

    {
    PROFILE_SCOPE(window_ordering_mst_build);
    
    if (!force_kappa_y0) {
        // Line 2: Window order ℵ (simplified for parallelism)
        auto window_order_start = std::chrono::high_resolution_clock::now();
        // Call window_order but get only the first entry (MST built inside)
        auto [first_entry_result, mst_tree_result, cached_distances_to_y0_result, mst_edge_distances_result] = window_order_first(Gx, X_vectors, y0, vec_y0);
        first_entry = first_entry_result;
        mst_tree = mst_tree_result;
        cached_distances_to_y0 = cached_distances_to_y0_result;
        mst_edge_distances = mst_edge_distances_result;
        auto window_order_end = std::chrono::high_resolution_clock::now();
        window_order_duration = std::chrono::duration_cast<std::chrono::microseconds>(window_order_end - window_order_start);
        std::cout << "Window order computation time: " << std::fixed << std::setprecision(3) << (window_order_duration.count() / 1000.0) << " ms" << std::endl;
    } else {
        std::cout << "Force kappa y0 mode: Skipping MST computation, processing queries in ID order" << std::endl;
    }
    }
    
        // Count total query vectors to process
        int total_query_vectors = 0;
        for (int x_node : Gx.get_nodes()) {
            if (X_vectors.count(x_node) > 0 && !X_vectors.at(x_node).empty()) {
                total_query_vectors++;
            }
        }
    // }
    
    // Shared data structures
    std::mutex queue_mutex;
    std::deque<std::pair<int, int>> work_queue; // [kappa_i, x_i] queue
    std::unordered_map<int, bool> processed_nodes;
    std::mutex processed_mutex;
    std::unordered_map<int, std::unordered_set<int>> windows;  // J_kappa cache
    
    // Termination tracking
    std::atomic<int> processed_count{0};
    std::atomic<int> active_threads{0};
    std::atomic<bool> all_work_done{false};
    
    // Progress tracking for pairs count
    std::atomic<int> total_pairs{0};
    
    
    // Progress tracking
    std::chrono::high_resolution_clock::time_point join_start_time = std::chrono::high_resolution_clock::now();
    std::mutex progress_mutex;
    
    // Set termination flag when all work is done
    auto check_termination = [&]() {
        if (processed_count >= total_query_vectors) {
            bool queue_empty = false;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                queue_empty = work_queue.empty();
            }
            if (queue_empty) {
                all_work_done = true;
            }
        }
    };
    
    // Initialize windows cache - y0 is not in X, so we don't add it to windows    
    // Initialize queue based on mode

    if (force_kappa_y0) {
        // Force all nodes to use y0 as parent (kappa_i = -1)
        std::cout << "Forcing kappa_i = y0 for all nodes (no seed inheritance from parent nodes)" << std::endl;
        for (int x_node : Gx.get_nodes()) {
            if (X_vectors.count(x_node) > 0 && !X_vectors.at(x_node).empty()) {
                work_queue.push_back({-1, x_node});  // Use -1 to indicate y0 as parent
            }
        }
    } else {
        // Normal behavior: use MST-based parent-child relationships
        if (first_entry.first != -1 && first_entry.second != -1) {
            // For the first entry (closest to y0), set kappa_i = -1 to indicate y0 as parent
            work_queue.push_back({-1, first_entry.second});  // {-1, x_i} for first entry
        }
        // Push all nodes to the back as safety fallback
        for (int x_node : Gx.get_nodes()) {
            if (X_vectors.count(x_node) > 0 && !X_vectors.at(x_node).empty()) {
                work_queue.push_back({-1, x_node});  // Use -1 to indicate y0 as parent (safety)
            }
        }
    }
    
    // Worker function for each thread
    auto worker = [&](int thread_id) {
        active_threads++;
        
        while (!all_work_done && !g_shutdown_requested) {
            std::pair<int, int> task;
            bool got_task = false;
            
            // Try to get a task from queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (!work_queue.empty()) {
                    task = work_queue.front();
                    work_queue.pop_front();
                    got_task = true;
                }
            }
            
            if (!got_task) {
                // Check termination condition
                check_termination();
                
                if (all_work_done || g_shutdown_requested) {
                    break;
                }
                
                // Not done yet, yield and try again
                std::this_thread::yield();
                continue;
            }
            
            int kappa_i = task.first;
            int x_i = task.second;
            
            // Debug: Print kappa_i value when force_kappa_y0 is true
            if (force_kappa_y0) {
                //std::cout << "DEBUG: Processing x_i=" << x_i << " with kappa_i=" << kappa_i << " (force_kappa_y0=true)" << std::endl;
                // Force kappa_i to -1 when force_kappa_y0 is true
                kappa_i = -1;
                //std::cout << "DEBUG: Overriding kappa_i to -1 for x_i=" << x_i << std::endl;
            }
            
            // Check if x_i already processed
            bool should_continue = false;
            {
                std::lock_guard<std::mutex> lock(processed_mutex);
                if (processed_nodes.count(x_i) > 0) {
                    should_continue = true; // Skip if already processed
                } else {
                    processed_nodes[x_i] = true;
                    processed_count++;
                }
            }
            
            if (should_continue) {
                continue;
            }
            
            const std::vector<float>& x_i_vec = X_vectors.at(x_i);
            if (x_i_vec.empty()) {
                continue;
            }
            
            // Check for shutdown before starting join processing
            if (g_shutdown_requested) {
                std::cout << "Shutdown requested during join processing, finishing current work..." << std::endl;
                break;
            }
            
            //kappa_i = -1; //XXX
            // Run join_slide for this [kappa_i, x_i] following SimJoin paper
            std::unordered_set<int> J_i;
            int seed_distance_count = 0;      // Counter for line 676 (seed distance calculations)
            int neighbor_distance_count = 0;  // Counter for line 719 (neighbor iteration distance calculations)
            int bfs_distance_count = 0;       // Counter for line 782 (BFS expansion distance calculations)
            
            // Check if this is the specific case we're looking for
            bool is_target_case = false; //(x_i == 1265); // (x_i == 263); //(x_i == 9573); //(x_i == 5086); //(x_i == 2017 || x_i == 2688);
            if (is_target_case) {
                std::cout << "*** FOUND TARGET CASE: kappa_i=" << kappa_i << ", x_i=" << x_i << " ***" << std::endl;
                
                // Compute distance between x_i vector and kappa_i vector
                if (kappa_i != -1) {
                    try {
                        const std::vector<float>& kappa_vec = X_vectors.at(kappa_i);
                        if (!kappa_vec.empty() && !x_i_vec.empty()) {
                            double dist_x_kappa = l2_distance(x_i_vec, kappa_vec);
                            std::cout << "  Distance between x_i=" << x_i << " and kappa_i=" << kappa_i << ": " << dist_x_kappa << std::endl;
                            
                            // Compare with stored MST edge distance
                            try {
                                double stored_dist = mst_edge_distances.at({kappa_i, x_i});
                                std::cout << "  Stored MST edge distance: " << stored_dist << std::endl;
                                double diff = std::abs(dist_x_kappa - stored_dist);
                                std::cout << "  Difference: " << diff << " (tolerance: 1e-10)" << std::endl;
                                if (diff > 1e-10) {
                                    std::cout << "  WARNING: Computed and stored distances differ significantly!" << std::endl;
                                } else {
                                    std::cout << "  ✓ Computed and stored distances match" << std::endl;
                                }
                            } catch (const std::out_of_range&) {
                                std::cout << "  Stored MST edge distance: NOT FOUND in mst_edge_distances" << std::endl;
                            }
                        } else {
                            std::cout << "  Cannot compute distance: kappa_vec or x_i_vec is empty" << std::endl;
                        }
                    } catch (const std::out_of_range&) {
                        std::cout << "  Cannot compute distance: kappa_i=" << kappa_i << " not found in X_vectors" << std::endl;
                    }
                } else {
                    std::cout << "  Cannot compute distance: kappa_i=-1 (using y0 as parent)" << std::endl;
                }
            }

            std::unordered_set<int> closest_node_for_query;
            auto query_start_time = std::chrono::high_resolution_clock::now();
            auto query_end_time  = std::chrono::high_resolution_clock::now();
            
            {

            PROFILE_SCOPE(simjoin_call);
            // Start timing for join_slide execution only
            query_start_time = std::chrono::high_resolution_clock::now();
            
            // Variable to store closest node for this query
            
            bool is_ood_query = false;
            if (topK > 0) {
                if (ood_flags[x_i+seed_offset]) {
                    // std::cout << "query is ood" << std::endl;
                    is_ood_query = true;
                } else {
                    // std::cout << "query is id" << std::endl;
                    is_ood_query = false;
                }
            }
            
            //if (kappa_i == -1 or seed_offset > 0) {
            if (kappa_i == -1) {
                if (is_ood_query) {
                    J_i = greedy_top_k_search(Gy, x_i, x_i_vec, Y_vectors, topK, patience, epsilon, seed_offset, bfs_distance_count);
                } else {
                    // Use y0 as parent (y0 is not in X, so we use y0_from_Y for the actual Y node)
                    J_i = join_slide(Gy, std::vector<float>(), x_i_vec, {y0_from_Y}, epsilon, w_queue, Y_vectors, 
                                seed_distance_count, neighbor_distance_count, bfs_distance_count, early_terminate, enable_top1_detection, enable_closest_fallback, is_target_case,
                                use_triangle_inequality, clusters2_centroids, clusters2_centroid_distances, global_to_clusters2_mapping, global_vectors, get_query_cache(x_i), local_to_global_mapping, cluster_id, x_i, enable_query_to_data_edges, k_top_data_points, data_offset,
                                break_before_bfs, sort_jj_by_distance, enable_seed_offset_filtering ? seed_offset : 0, cache_closest_only ? &closest_node_for_query : nullptr,
                                collect_bfs_data ? &g_greedy_visited[x_i] : nullptr,
                                collect_bfs_data ? &g_bfs_seeds[x_i] : nullptr, 
                                collect_bfs_data ? &g_bfs_visited[x_i] : nullptr,
                                collect_bfs_data ? &g_final_join_results[x_i] : nullptr,
                                max_bfs_out_of_range_tolerance,
                                adaptive_bfs_threshold_factor,
                                one_hop_data_only,
                                number_cached
                                );
                }
            } else {
                //assert(!force_kappa_y0);
                // Use kappa_i as parent
                const std::vector<float>& kappa_vec = X_vectors.at(kappa_i);
                if (!kappa_vec.empty()) {
                    try {
                        const auto& window = windows.at(kappa_i);
                        if (!window.empty()) {
                            /*std::unordered_set<int> seeds = window;
                            if (seed_offset > 0) {
                                seeds.insert(x_i + seed_offset);
                                if (one_hop_data_only)
                                    seeds.insert(y0_from_Y);
                            }*/
                            if (is_ood_query) {
                                J_i = greedy_top_k_search(Gy, x_i, x_i_vec, Y_vectors, topK, patience, epsilon, seed_offset, bfs_distance_count);
                            } else {
                                J_i = join_slide(Gy, kappa_vec, x_i_vec, window, epsilon, w_queue, Y_vectors, 
                                                seed_distance_count, neighbor_distance_count, bfs_distance_count, early_terminate, enable_top1_detection, enable_closest_fallback, is_target_case,
                                                use_triangle_inequality, clusters2_centroids, clusters2_centroid_distances, global_to_clusters2_mapping, global_vectors, get_query_cache(x_i), local_to_global_mapping, cluster_id, x_i, enable_query_to_data_edges, k_top_data_points, data_offset,
                                                break_before_bfs, sort_jj_by_distance, enable_seed_offset_filtering ? seed_offset : 0, cache_closest_only ? &closest_node_for_query : nullptr,
                                                collect_bfs_data ? &g_greedy_visited[x_i] : nullptr,
                                                collect_bfs_data ? &g_bfs_seeds[x_i] : nullptr, 
                                                collect_bfs_data ? &g_bfs_visited[x_i] : nullptr,
                                                collect_bfs_data ? &g_final_join_results[x_i] : nullptr,
                                                max_bfs_out_of_range_tolerance,
                                                adaptive_bfs_threshold_factor,
                                                one_hop_data_only,
                                                number_cached
                                                );
                            }
                        } else {
                            // Empty window, use y0
                            if (is_ood_query) {
                                J_i = greedy_top_k_search(Gy, x_i, x_i_vec, Y_vectors, topK, patience, epsilon, seed_offset, bfs_distance_count);
                            } else {
                                // Empty window, use y0
                                J_i = join_slide(Gy, std::vector<float>(), x_i_vec, {y0_from_Y}, epsilon, w_queue, Y_vectors, 
                                                seed_distance_count, neighbor_distance_count, bfs_distance_count, early_terminate, enable_top1_detection, enable_closest_fallback, is_target_case,
                                                use_triangle_inequality, clusters2_centroids, clusters2_centroid_distances, global_to_clusters2_mapping, global_vectors, get_query_cache(x_i), local_to_global_mapping, cluster_id, x_i, enable_query_to_data_edges, k_top_data_points, data_offset,
                                                break_before_bfs, sort_jj_by_distance, enable_seed_offset_filtering ? seed_offset : 0, cache_closest_only ? &closest_node_for_query : nullptr,
                                                collect_bfs_data ? &g_greedy_visited[x_i] : nullptr,
                                                collect_bfs_data ? &g_bfs_seeds[x_i] : nullptr, 
                                                collect_bfs_data ? &g_bfs_visited[x_i] : nullptr,
                                                collect_bfs_data ? &g_final_join_results[x_i] : nullptr,
                                                max_bfs_out_of_range_tolerance,
                                                adaptive_bfs_threshold_factor,
                                                one_hop_data_only,
                                                number_cached
                                                );
                            }
                        }
                    } catch (const std::out_of_range&) {
                        // Window not found, use y0
                        if (is_ood_query) {
                            J_i = greedy_top_k_search(Gy, x_i, x_i_vec, Y_vectors, topK, patience, epsilon, seed_offset, bfs_distance_count);
                        } else {
                            J_i = join_slide(Gy, std::vector<float>(), x_i_vec, {y0_from_Y}, epsilon, w_queue, Y_vectors, 
                                            seed_distance_count, neighbor_distance_count, bfs_distance_count, early_terminate, enable_top1_detection, enable_closest_fallback, is_target_case,
                                            use_triangle_inequality, clusters2_centroids, clusters2_centroid_distances, global_to_clusters2_mapping, global_vectors, get_query_cache(x_i), local_to_global_mapping, cluster_id, x_i, enable_query_to_data_edges, k_top_data_points, data_offset,
                                            break_before_bfs, sort_jj_by_distance, enable_seed_offset_filtering ? seed_offset : 0, cache_closest_only ? &closest_node_for_query : nullptr,
                                            collect_bfs_data ? &g_greedy_visited[x_i] : nullptr,
                                            collect_bfs_data ? &g_bfs_seeds[x_i] : nullptr, 
                                            collect_bfs_data ? &g_bfs_visited[x_i] : nullptr,
                                            collect_bfs_data ? &g_final_join_results[x_i] : nullptr,
                                            max_bfs_out_of_range_tolerance,
                                            adaptive_bfs_threshold_factor,
                                            one_hop_data_only,
                                            number_cached
                                            );
                                        }
                    }
                } else {
                    assert(false && "kappa_i vector is empty");
                }
            }

            query_end_time = std::chrono::high_resolution_clock::now();

            }

            {

            PROFILE_SCOPE(store_join_window);
            
            // Store the join window for x_i (following SimJoin paper)
            if (!force_kappa_y0) {
                std::lock_guard<std::mutex> lock(processed_mutex);
                
                // If we have a closest node, use only that node
                if (cache_closest_only && enable_closest_fallback && !closest_node_for_query.empty()) {
                    windows[x_i] = closest_node_for_query;
                    
                    // If in multi-cluster mode, also store in global variable
                    if (cluster_id >= 0) {
                        // Ensure global variable has enough space
                        if (g_cluster_windows.size() <= static_cast<size_t>(cluster_id)) {
                            g_cluster_windows.resize(cluster_id + 1);
                        }
                        g_cluster_windows[cluster_id][x_i] = closest_node_for_query;
                    }
                } else {
                    windows[x_i] = J_i;
                    
                    // If in multi-cluster mode, also store in global variable
                    if (cluster_id >= 0) {
                        // Ensure global variable has enough space
                        if (g_cluster_windows.size() <= static_cast<size_t>(cluster_id)) {
                            g_cluster_windows.resize(cluster_id + 1);
                        }
                        g_cluster_windows[cluster_id][x_i] = J_i;
                    }
                }
            }

            }

            std::vector<std::pair<int, double>> matches_with_distances;
            {
            
                PROFILE_SCOPE(result_writing);
            // Store join window size and parent information for visualization
            if (true) //XXX
            {
                std::lock_guard<std::mutex> lock(processed_mutex);
                
                // DEBUG: Add comprehensive assertions to identify MST property violation
                //std::cout << "\n=== MST Property Debug for x_i=" << x_i << " ===" << std::endl;
                
                // 1. Verify x_i exists in X_vectors
                assert(X_vectors.count(x_i) > 0 && "x_i not found in X_vectors");
                assert(!X_vectors.at(x_i).empty() && "x_i vector is empty");
                
                // 2. Verify kappa_i exists in X_vectors (if not -1)
                if (kappa_i != -1) {
                    assert(X_vectors.count(kappa_i) > 0 && "kappa_i not found in X_vectors");
                    assert(!X_vectors.at(kappa_i).empty() && "kappa_i vector is empty");
                    
                    // 3. Verify kappa_i exists in MST tree
                    assert(mst_tree.count(kappa_i) > 0 && "kappa_i not found in MST tree");
                    
                    // 4. Verify x_i is a child of kappa_i in MST
                    const auto& kappa_children = mst_tree.at(kappa_i);
                    bool x_i_is_child = false;
                    for (int child : kappa_children) {
                        if (child == x_i) {
                            x_i_is_child = true;
                            break;
                        }
                    }
                    assert(x_i_is_child && "x_i is not a child of kappa_i in MST");
                    
                    // 5. Verify MST edge distance exists
                    assert(mst_edge_distances.count({kappa_i, x_i}) > 0 && "MST edge distance not found for kappa_i->x_i");
                    double mst_edge_dist = mst_edge_distances.at({kappa_i, x_i});
                    //std::cout << "  MST edge distance (kappa_i->x_i): " << mst_edge_dist << std::endl;
                }
                
                // Reuse cached distances instead of recomputing
                double dist_to_kappa_i = 0.0;
                double dist_to_y0 = 0.0;
                
                // 6. Get cached distance to y0 and verify it exists
                try {
                    dist_to_y0 = cached_distances_to_y0.at(x_i);
                    //std::cout << "  Cached dist_to_y0: " << dist_to_y0 << std::endl;
                } catch (const std::out_of_range&) {
                    //if (seed_offset == 0) {
                    if (!force_kappa_y0) {
                        std::cout << "  ERROR: x_i not found in cached_distances_to_y0!" << std::endl;
                        assert(false && "x_i not found in cached_distances_to_y0");
                    }
                }
                
                // 7. Calculate distance to kappa_i and verify consistency
                if (kappa_i != -1) {
                    const std::vector<float>& kappa_vec = X_vectors.at(kappa_i);
                    if (!kappa_vec.empty()) {
                        dist_to_kappa_i = l2_distance(x_i_vec, kappa_vec);
                        //std::cout << "  Runtime dist_to_kappa_i: " << dist_to_kappa_i << std::endl;
                        
                        // 8. Verify consistency with MST edge distance
                        double mst_edge_dist = mst_edge_distances.at({kappa_i, x_i});
                        if (std::abs(dist_to_kappa_i - mst_edge_dist) > 1e-6) {
                            std::cout << "  ERROR: MST edge distance mismatch!" << std::endl;
                            std::cout << "    MST edge distance: " << mst_edge_dist << std::endl;
                            std::cout << "    Runtime distance: " << dist_to_kappa_i << std::endl;
                            std::cout << "    Difference: " << std::abs(dist_to_kappa_i - mst_edge_dist) << std::endl;
                            assert(false && "MST edge distance mismatch with runtime calculation");
                        }
                        
                        // 9. Basic validation (MST properties verified comprehensively after construction)
                        //std::cout << "  Basic validation:" << std::endl;
                        //std::cout << "    distance(x_i → kappa_i): " << dist_to_kappa_i << std::endl;
                        //std::cout << "    distance(x_i → y0): " << dist_to_y0 << std::endl;
                        //std::cout << "    MST edge distance consistency: ✓" << std::endl;
                        
                    } else {
                        std::cout << "  ERROR: kappa_i vector is empty!" << std::endl;
                        assert(false && "Could not find kappa_i in X_vectors");
                    }
                } else if (kappa_i == -1) {
                    // When kappa_i = -1, the node connects to y0, so just store dist_to_kappa_i as 0.0
                    dist_to_kappa_i = 0.0;
                    //std::cout << "  kappa_i = -1, connecting directly to y0" << std::endl;
                }
                
                // 10. Summary (MST properties verified comprehensively after construction)
                if (kappa_i != -1) {
                    //std::cout << "  Summary:" << std::endl;
                    //std::cout << "    distance(x_i → kappa_i): " << dist_to_kappa_i << std::endl;
                    //std::cout << "    distance(x_i → y0): " << dist_to_y0 << std::endl;
                    //std::cout << "    MST structure validated: ✓" << std::endl;
                    
                    if (dist_to_kappa_i >= dist_to_y0) {
                        std::cout << "LIMIT: kappa_i is farther than y0 for x_i=" << x_i << std::endl;
                        std::cout << "  kappa_i=" << kappa_i << " (not -1)" << std::endl;
                        std::cout << "    distance(x_i → kappa_i): " << dist_to_kappa_i << std::endl;
                        std::cout << "    distance(x_i → y0): " << dist_to_y0 << std::endl;
                        std::cout << "  Expected: distance(x_i → kappa_i) < distance(x_i → y0)" << std::endl;
                        
                        // 11. Additional debugging info
                        std::cout << "  Additional debug info:" << std::endl;
                        std::cout << "    x_i vector size: " << x_i_vec.size() << std::endl;
                        std::cout << "    kappa_i vector size: " << X_vectors.at(kappa_i).size() << std::endl;
                        std::cout << "    y0 vector size: " << vec_y0.size() << std::endl;
                        
                        //assert(false && "LIMIT: kappa_i is farther than y0 for x_i");
                    } else {
                        //std::cout << "  MST property verified successfully!" << std::endl;
                    }
                }
                
                //std::cout << "=== End MST Property Debug ===\n" << std::endl;
                
            }
            
            
            // Add thread-local distance computations to global counter
            g_distance_computations.fetch_add(thread_distance_computations, std::memory_order_relaxed);
            
            // Add the three separate distance computation counters to global counters
            g_seed_distance_computations.fetch_add(seed_distance_count, std::memory_order_relaxed);
            g_neighbor_distance_computations.fetch_add(neighbor_distance_count, std::memory_order_relaxed);
            g_bfs_distance_computations.fetch_add(bfs_distance_count, std::memory_order_relaxed);

            // Calculate distances for all matches and filter by epsilon
            
            int query_matches_filtered = 0;
            
            for (int y : J_i) {
                try {
                    const std::vector<float>& y_vec = Y_vectors.at(y);
                    if (!y_vec.empty()) {
                        double dist = l2_distance(x_i_vec, y_vec);
                        if (dist <= epsilon) { // when sort work sharing
                            bool include_match = true;
                            
                            // Filter by data_offset when query-to-data edges are enabled
                            if (enable_query_to_data_edges && y >= data_offset) {
                                include_match = false;
                            }
                            
                            // Filter by seed_offset when seed offset is enabled
                            if (seed_offset > 0 && y >= seed_offset) {
                                include_match = false;
                            }
                            
                            if (include_match) {
                                matches_with_distances.push_back({y, dist});
                            } else {
                                query_matches_filtered++;
                            }
                        }
                    }
                } catch (const std::out_of_range&) {
                    // Skip if y not found
                }
            }

            //std::cout << "DEBUG: join_slide returned J_i with " << J_i.size() << " matches" << " and " << matches_with_distances.size() << " matches with distance <= epsilon" << std::endl;

            // Sort by distance
            std::sort(matches_with_distances.begin(), matches_with_distances.end(),
                        [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                            return a.second < b.second;
                        });

            // Update total pairs count
            total_pairs.fetch_add(matches_with_distances.size());
            
            // Stream join output for this x_i
            if (!join_output_filename.empty()) {
                static std::mutex file_mutex;
                std::ofstream join_output_file(join_output_filename, std::ios::app);
                
                std::lock_guard<std::mutex> lock(file_mutex);


                if (join_output_file.is_open()) {
                    // Write output line: <x_i> <kappa_i> <num_join_results> <seed> <neighbor> <BFS> <processing_time_for_x_i> <memory_usage> [clusters_visited] <join_result_1> <join_result_2> ...
                    auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end_time - query_start_time);
                    double memory_usage = get_memory_usage_mb();
                    join_output_file << x_i << " " << kappa_i << " " << matches_with_distances.size() << " " 
                                   << static_cast<double>(seed_distance_count) << " " << static_cast<double>(neighbor_distance_count) << " " 
                                   << static_cast<double>(bfs_distance_count) << " " << static_cast<double>(query_duration.count()) / 1000.0 << " " << memory_usage;
                    
                    // Add clusters visited count when pruning is enabled
                    if (use_triangle_inequality) {
                        // Count clusters visited for this query (size of query_to_centroid_distances cache)
                        int clusters_visited = get_query_cache(x_i)->size();
                        join_output_file << " " << clusters_visited;
                    }
                    if (print_join_output) { // XXX don't print outputs to save disk space
                        for (const auto& [y_id, dist] : matches_with_distances) {
                            join_output_file << " " << y_id;
                        }
                    }
                    join_output_file << std::endl;
                    join_output_file.flush();
                }
            }

            }

            {

            PROFILE_SCOPE(add_query_to_data_edges);
            
            // Add query-to-data edges if enabled
            if (enable_query_to_data_edges && !matches_with_distances.empty()) {
                // Map query ID to data space by adding offset
                int query_in_data_space = data_offset + x_i;
                int edges_added = 0;
                
                // Add the query vector to Y_vectors so it can be processed during traversal
                Y_vectors[query_in_data_space] = x_i_vec;
                
                // Add edges between query and join results
                int query_to_data_edges_added = 0;
                int query_to_query_edges_added = 0;
                int data_to_data_edges_added = 0;
                
                if (k_top_data_points > 0) {
                    int k_to_use = std::min(k_top_data_points, static_cast<int>(matches_with_distances.size()));
                    
                    // Add edges between all pairs of top-k data points
                    for (int i = 0; i < k_to_use; i++) {
                        for (int j = i + 1; j < k_to_use; j++) {
                            int y1_id = matches_with_distances[i].first;
                            int y2_id = matches_with_distances[j].first;
                            double dist1 = matches_with_distances[i].second;
                            double dist2 = matches_with_distances[j].second;
                            
                            // Add bidirectional edge between the two data points
                            Gy.add_edge_online(y1_id, y2_id, 0.0);
                            Gy.add_edge_online(y2_id, y1_id, 0.0);
                            edges_added++;
                            data_to_data_edges_added++; // Both are data points
                        }
                    }
                } else {
                    // Add edges to all join results (original behavior)
                    for (const auto& [y_id, dist] : matches_with_distances) {
                        // Add edge from query to data/query node
                        Gy.add_edge_online(query_in_data_space, y_id, dist);
                        // Add edge from data/query node to query (bidirectional)
                        Gy.add_edge_online(y_id, query_in_data_space, dist);
                        edges_added++;
                        
                        // Count query-to-data vs query-to-query edges
                        if (y_id >= data_offset) {
                            // Both endpoints are in query space (>= data_offset)
                            query_to_query_edges_added++;
                        } else {
                            // One endpoint is query space, other is data space
                            query_to_data_edges_added++;
                        }
                    }
                }
                
                // Update global counters
                g_query_to_data_edges += query_to_data_edges_added;
                g_query_to_query_edges += query_to_query_edges_added;
                g_data_to_data_edges += data_to_data_edges_added;
                
                if (edges_added > 0) {
                    std::cout << "  Added " << edges_added << " query-to-data/query edges for query " << x_i 
                              << " (mapped to data space ID " << query_in_data_space << ") and added query vector to Y_vectors" << std::endl;
                    
                    // Verify that the added edges can be retrieved via get_neighbors
                    std::cout << "  Verifying edge retrieval for query " << x_i << " (ID " << query_in_data_space << "):" << std::endl;
                    const auto& neighbors = Gy.get_neighbors(query_in_data_space);
                    std::cout << "    Found " << neighbors.size() << " neighbors via get_neighbors" << std::endl;
                    
                    // Check if our added edges are in the retrieved neighbors
                    int verified_edges = 0;
                    for (const auto& [y_id, dist] : matches_with_distances) {
                        bool found_as_neighbor = false;
                        for (const auto& [neighbor_id, neighbor_dist] : neighbors) {
                            if (neighbor_id == y_id) {
                                found_as_neighbor = true;
                                verified_edges++;
                                break;
                            }
                        }
                        if (!found_as_neighbor && k_top_data_points == 0) {
                            std::cout << "    WARNING: Edge to " << y_id << " not found in get_neighbors!" << std::endl;
                        }
                    }
                    std::cout << "    Verified " << verified_edges << "/" << matches_with_distances.size() << " edges are retrievable" << std::endl;
                }
            }
            
            // If in multi-cluster mode, store processing time for this query
            if (cluster_id >= 0) {
                auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end_time - query_start_time);
                
                // Ensure global variable has enough space
                if (g_cluster_processing_times.size() <= static_cast<size_t>(cluster_id)) {
                    g_cluster_processing_times.resize(cluster_id + 1);
                }
                g_cluster_processing_times[cluster_id][x_i] = static_cast<double>(query_duration.count()) / 1000.0; // Convert microseconds to milliseconds with sub-ms precision
            }

            thread_distance_computations = 0;  // Reset for next query 
            
            // Push next work entries for children of x_i in the MST
            {
                std::scoped_lock lock(queue_mutex, processed_mutex);  // Avoid deadlock
                // Find all children of x_i in the MST tree
                try {
                    const std::vector<int>& children = mst_tree.at(x_i);
                    for (int child : children) {
                        // Check if child is already processed (atomic with queue push)
                        bool already_processed = (processed_nodes.count(child) > 0);
                        
                        if (!already_processed && 
                            X_vectors.count(child) > 0 && 
                            !X_vectors.at(child).empty()) {
                            if (force_kappa_y0) {
                                // Force mode: all children use y0 as parent (kappa_i = -1)
                                work_queue.push_front({-1, child});
                            } else {
                                // Normal mode: use MST-based parent-child relationships
                                work_queue.push_front({x_i, child});  // Push to front for priority
                            }
                        }
                    }
                } catch (const std::out_of_range&) {
                    // No children found, skip
                }
            }
            
            // Print progress only when processed_count is a multiple of 100
            if (cluster_id < 0) {
                std::lock_guard<std::mutex> lock(progress_mutex);
                if (processed_count % 100 == 0) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(current_time - join_start_time);
                    
                    std::cout << "Progress: " << processed_count << "/" << total_query_vectors 
                             << " queries processed, " << total_pairs.load() << " pairs found, "
                             << g_distance_computations.load() << " distance computations, "
                             << std::fixed << std::setprecision(3) << (elapsed.count() / 1000.0) << " ms elapsed";
                    
                    // Check for shutdown request during progress reporting
                    if (g_shutdown_requested) {
                        std::cout << " [SHUTDOWN REQUESTED]" << std::endl;
                    } else {
                        std::cout << std::endl;
                    }
                    
                    // Stream progress to file
                    // Remove existing progress file to avoid appending to old results
                    std::remove((result_dir + "/progress.txt").c_str());
                    static std::ofstream progress_file(result_dir + "/progress.txt");
                    if (progress_file.is_open()) {
                        double memory_usage = get_memory_usage_mb();
                        progress_file << processed_count << " " << total_pairs.load() << " " 
                                    << static_cast<double>(g_seed_distance_computations.load()) << " " 
                                    << static_cast<double>(g_neighbor_distance_computations.load()) << " " 
                                    << static_cast<double>(g_bfs_distance_computations.load()) << " " 
                                    << std::fixed << std::setprecision(6) << (elapsed.count() / 1000.0) << " " << memory_usage << std::endl;
                        progress_file.flush();  // Ensure immediate write
                    }
                }
            }
            }
        }
        
        active_threads--;
    };

    {
        PROFILE_SCOPE(result_writing_2);
    
    // Launch worker threads
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker, i);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Separate BFS loop for data collection (if enabled) - AFTER all join_slide calls finish
    if (break_before_bfs) {
        std::cout << "Performing BFS expansion for data collection..." << std::endl;
        std::cout << "Total pairs found before BFS: " << total_pairs.load() << std::endl;
        
        // Reset total_pairs to 0 since we counted only seeds before, now we'll count the full BFS results
        total_pairs.store(0);
        
        // Process each query for BFS
        for (int x_i = 0; x_i < static_cast<int>(X_vectors.size()); x_i++) {
            try {
                const std::vector<float>& x_i_vec = X_vectors.at(x_i);
                if (x_i_vec.empty()) {
                    continue;
                }
                
                // Get the seeds collected during join_slide
                std::unordered_set<int> seeds(g_bfs_seeds[x_i].begin(), g_bfs_seeds[x_i].end());
                
                // Determine the parent vector for BFS (same logic as original)
                std::vector<float> parent_vec;
                int kappa_i = -1;
                
                // Find kappa_i from the MST or use y0
                if (!force_kappa_y0) {
                    // Use MST-based parent-child relationships
                    for (const auto& [parent, children] : mst_tree) {
                        for (int child : children) {
                            if (child == x_i) {
                                kappa_i = parent;
                                break;
                            }
                        }
                        if (kappa_i != -1) break;
                    }
                }
                
                if (kappa_i == -1) {
                    // Use y0 as parent, but we need the actual parent vector
                    // For BFS, we use the query vector as the reference
                    parent_vec = x_i_vec;
                } else {
                    // Use kappa_i as parent
                    const std::vector<float>& kappa_vec = X_vectors.at(kappa_i);
                    parent_vec = kappa_vec;
                }
                
                // Perform BFS with depth collection
                int bfs_distance_count = 0;
                std::unordered_set<int> J_i = perform_bfs_with_depth_collection(Gy, x_i_vec, parent_vec, seeds, epsilon, Y_vectors,
                    bfs_distance_count, use_triangle_inequality, clusters2_centroids, clusters2_centroid_distances,
                    global_to_clusters2_mapping, global_vectors, get_query_cache(x_i), local_to_global_mapping,
                    cluster_id, x_i, enable_query_to_data_edges, data_offset, enable_closest_fallback, enable_seed_offset_filtering ? seed_offset : 0,
                    sort_jj_by_distance, nullptr, // closest_node_output not needed here
                    collect_bfs_data ? &g_bfs_visited[x_i] : nullptr, 
                    collect_bfs_data ? &g_final_join_results[x_i] : nullptr,
                    max_bfs_out_of_range_tolerance,
                    adaptive_bfs_threshold_factor);
                
                // Update the final results and count pairs
                {
                    std::lock_guard<std::mutex> lock(processed_mutex);
                    if (!force_kappa_y0) {
                        windows[x_i] = J_i;
                    }
                    // Count the pairs found in this BFS expansion
                    total_pairs.fetch_add(J_i.size()); // here, this can be larger than the actual... when enable_closest_fallback is enabled
                }
                
            } catch (const std::out_of_range&) {
                // Skip if x_i not found
            }
        }
        
        std::cout << "BFS expansion completed for all queries" << std::endl;
    }
    
    // Measure total join time
    auto join_end_time = std::chrono::high_resolution_clock::now();
    auto join_duration = std::chrono::duration_cast<std::chrono::microseconds>(join_end_time - join_start_time);
    std::cout << "Join computation time: " << std::fixed << std::setprecision(3) << (join_duration.count() / 1000.0) << " ms" << std::endl;
    std::cout << "Total pairs found: " << total_pairs.load() << std::endl;
    
    // Print query-to-data and query-to-query edges summary if enabled
    if (enable_query_to_data_edges) {
        std::cout << "Query-to-data edges optimization enabled - edges added to data index for future traversal" << std::endl;
        std::cout << "Total query-to-data edges created: " << g_query_to_data_edges.load() << std::endl;
        std::cout << "Total query-to-data edges traversed: " << g_query_to_data_edges_traversed.load() << std::endl;
        std::cout << "Total query-to-query edges created: " << g_query_to_query_edges.load() << std::endl;
        std::cout << "Total query-to-query edges traversed: " << g_query_to_query_edges_traversed.load() << std::endl;
        std::cout << "Total data-to-data edges created: " << g_data_to_data_edges.load() << std::endl;
        std::cout << "Total data-to-data edges traversed: " << g_data_to_data_edges_traversed.load() << std::endl;
    }
    
    // Check if shutdown was requested
    if (g_shutdown_requested) {
        std::cout << "Program was interrupted, saving partial results..." << std::endl;
        // Save partial profiling results immediately
        std::string partial_profiling_filename = "partial_profiling_results.txt";
        g_profile_data.save_stats_to_file(partial_profiling_filename);
    }
    
    // Print timing summary
    std::cout << "\n=== Timing Summary ===" << std::endl;
    std::cout << "Window order (includes MST): " << std::fixed << std::setprecision(3) << (window_order_duration.count() / 1000.0) << " ms" << std::endl;
    std::cout << "Join computation: " << std::fixed << std::setprecision(3) << (join_duration.count() / 1000.0) << " ms" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(3) << ((window_order_duration.count() + join_duration.count()) / 1000.0) << " ms" << std::endl;
    std::cout << "=====================" << std::endl;

}
    

}

// ----------------------------------------------------------------------
// Connectivity checking utilities
// ----------------------------------------------------------------------

/**
 * Check connectivity of a graph using BFS
 * @param graph The graph to check
 * @param graph_name Name of the graph for logging
 * @return true if graph is connected, false otherwise
 */
bool check_graph_connectivity(const Graph& graph, const std::string& graph_name) {
    if (graph.size() == 0) {
        std::cout << "  " << graph_name << ": Empty graph (0 nodes)" << std::endl;
        return true;
    }
    
    auto nodes = graph.get_nodes();
    if (nodes.empty()) {
        std::cout << "  " << graph_name << ": No nodes in graph" << std::endl;
        return true;
    }
    
    // Start BFS from first node
    std::unordered_set<int> visited;
    std::queue<int> bfs_queue;
    
    int start_node = nodes[0];
    bfs_queue.push(start_node);
    visited.insert(start_node);
    
    while (!bfs_queue.empty()) {
        int current = bfs_queue.front();
        bfs_queue.pop();
        
        const auto& neighbors = graph.get_neighbors(current);
        for (const auto& [neighbor, weight] : neighbors) {
            if (visited.count(neighbor) == 0) {
                visited.insert(neighbor);
                bfs_queue.push(neighbor);
            }
        }
    }
    
    bool is_connected = (visited.size() == nodes.size());
    
    std::cout << "  " << graph_name << " connectivity check:" << std::endl;
    std::cout << "    Total nodes: " << nodes.size() << std::endl;
    std::cout << "    Visited nodes: " << visited.size() << std::endl;
    std::cout << "    Connected: " << (is_connected ? "YES" : "NO") << std::endl;
    
    if (!is_connected) {
        std::cout << "    WARNING: Graph is not fully connected!" << std::endl;
        std::cout << "    Unvisited nodes: ";
        for (int node : nodes) {
            if (visited.count(node) == 0) {
                std::cout << node << " ";
            }
        }
        std::cout << std::endl;
    }
    
    return is_connected;
}

/**
 * Check connectivity of query index after loading
 * @param Gx Query graph (supplier)
 * @param X_vectors Query vectors
 */
void check_query_connectivity_after_loading(const Graph& Gx, 
                                           const std::unordered_map<int, std::vector<float>>& X_vectors) {
    std::cout << "\n=== QUERY CONNECTIVITY CHECK AFTER LOADING ===" << std::endl;
    
    // Check query graph (Gx) connectivity
    bool query_connected = check_graph_connectivity(Gx, "Query Graph (Gx)");
    
    // Check query vector data consistency
    std::cout << "  Query vector data consistency:" << std::endl;
    std::cout << "    Query vectors: " << X_vectors.size() << " vectors" << std::endl;
    
    // Check for empty query vectors
    int empty_query_vectors = 0;
    for (const auto& [id, vec] : X_vectors) {
        if (vec.empty()) {
            empty_query_vectors++;
        }
    }
    
    std::cout << "    Empty query vectors: " << empty_query_vectors << std::endl;
    
    // Check query ID range
    if (!X_vectors.empty()) {
        int min_query_id = INT_MAX, max_query_id = INT_MIN;
        for (const auto& [id, _] : X_vectors) {
            min_query_id = std::min(min_query_id, id);
            max_query_id = std::max(max_query_id, id);
        }
        std::cout << "    Query ID range: " << min_query_id << " to " << max_query_id << std::endl;
    }
    
    // Summary
    std::cout << "  Summary:" << std::endl;
    std::cout << "    Query graph connected: " << (query_connected ? "YES" : "NO") << std::endl;
    
    if (!query_connected) {
        std::cout << "  WARNING: Query graph is not fully connected!" << std::endl;
        std::cout << "  This may affect algorithm performance and correctness." << std::endl;
    } else {
        std::cout << "  Query graph is fully connected." << std::endl;
    }
    
    std::cout << "=== END QUERY CONNECTIVITY CHECK ===\n" << std::endl;
}

// ----------------------------------------------------------------------
// Data loading utilities
// ----------------------------------------------------------------------

/**
 * Load CAGRA data from JSON file and convert to vectors and graph
 */
std::pair<std::unordered_map<int, std::vector<float>>, std::unordered_map<int, std::vector<int>>> 
load_cagra_data(const std::string& json_file) {
    std::cout << "Loading " << json_file << "..." << std::endl;
    
    std::ifstream file(json_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + json_file);
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    file.close();
    
    JsonReader reader;
    JsonValue root = reader.parse(json_content);
    
    std::unordered_map<int, std::vector<float>> vectors;
    std::unordered_map<int, std::vector<int>> graph_edges;
    
    // Handle different JSON structures
    if (root["index_info"].isObject()) {
        // Full JSON structure
        const JsonValue& vectors_data = root["vectors"];
        for (size_t i = 0; i < vectors_data.size(); ++i) {
            const JsonValue& item = vectors_data[i];
            int vector_id = item["id"].asInt();
            const JsonValue& neighbors = item["neighbors"];
            
            vectors[vector_id] = std::vector<float>();  // Placeholder
            for (size_t j = 0; j < neighbors.size(); ++j) {
                graph_edges[vector_id].push_back(neighbors[j].asInt());
            }
        }
    } else {
        // Simple list structure
        for (size_t i = 0; i < root.size(); ++i) {
            const JsonValue& item = root[i];
            int vector_id = item["id"].asInt();
            const JsonValue& neighbors = item["neighbors"];
            
            vectors[vector_id] = std::vector<float>();  // Placeholder
            for (size_t j = 0; j < neighbors.size(); ++j) {
                graph_edges[vector_id].push_back(neighbors[j].asInt());
            }
        }
    }
    
    std::cout << "Loaded " << vectors.size() << " vectors with neighbors" << std::endl;
    return {vectors, graph_edges};
}

/**
 * Load actual embedding vectors from fvecs file
 */
std::unordered_map<int, std::vector<float>> 
load_embedding_vectors(const std::string& embedding_file, int num_vectors, int dim) {
    std::cout << "Loading embedding vectors from fvecs file: " << embedding_file << "..." << std::endl;
    
    std::unordered_map<int, std::vector<float>> vectors;
    std::ifstream file(embedding_file, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open embedding file: " + embedding_file);
    }
    
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
        if (vec_dim != dim) {
            std::cout << "Warning: Vector " << i << " has dimension " << vec_dim << ", expected " << dim << std::endl;
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
    
    std::cout << "Loaded " << vectors.size() << " vectors from fvecs file" << std::endl;
    return vectors;
}

/**
 * Build Graph object from neighbors dictionary with computed distances (for X graph MST)
 */
Graph build_graph_from_neighbors_with_distances(
    const std::unordered_map<int, std::vector<int>>& neighbors_dict,
    const std::unordered_map<int, std::vector<float>>& vectors_dict) {
    
    std::cout << "Building graph from neighbors with distances..." << std::endl;
    Graph graph;
    
    int valid_vectors = 0;
    int edges_added = 0;
    int total_neighbors_processed = 0;
    
    for (const auto& [vector_id, neighbors] : neighbors_dict) {
        try {
            const std::vector<float>& vec = vectors_dict.at(vector_id);
            if (!vec.empty()) {
                valid_vectors++;
                for (int neighbor_id : neighbors) {
                    total_neighbors_processed++;
                    try {
                        const std::vector<float>& neighbor_vec = vectors_dict.at(neighbor_id);
                        if (!neighbor_vec.empty()) {
                            // Compute distance for MST on X graph
                            double dist = l2_distance(vec, neighbor_vec);
                            graph.add_edge(vector_id, neighbor_id, dist);
                            edges_added++;
                        }
                    } catch (const std::out_of_range&) {
                        // Neighbor not found in embedding data, but still add edge with placeholder weight
                        graph.add_edge(vector_id, neighbor_id, 1.0);  // Placeholder weight
                        edges_added++;
                    }
                }
            }
        } catch (const std::out_of_range&) {
            // Vector not found, skip
        }
    }

    //std::cout << boost::stacktrace::stacktrace() << std::endl;
    
    std::cout << "Built graph with " << graph.size() << " nodes, " 
              << valid_vectors << " valid vectors, " << edges_added << " edges" << std::endl;
    std::cout << "Total neighbors processed: " << total_neighbors_processed << std::endl;
    return graph;
}



/**
 * Build Graph object from neighbors dictionary without computing distances (for Y graph)
 */
Graph build_graph_from_neighbors_simple(
    const std::unordered_map<int, std::vector<int>>& neighbors_dict,
    const std::unordered_map<int, std::vector<float>>& vectors_dict) {
    
    std::cout << "Building graph from neighbors without distances..." << std::endl;
    Graph graph;
    
    int valid_vectors = 0;
    int edges_added = 0;
    
    for (const auto& [vector_id, neighbors] : neighbors_dict) {
        try {
            const std::vector<float>& vec = vectors_dict.at(vector_id);
            if (!vec.empty()) {
                valid_vectors++;
                for (int neighbor_id : neighbors) {
                    try {
                        const std::vector<float>& neighbor_vec = vectors_dict.at(neighbor_id);
                        if (!neighbor_vec.empty()) {
                            // Add edge without computing distance (SimJoin computes distances on-the-fly)
                            graph.add_edge(vector_id, neighbor_id, 0.0);  // Placeholder weight
                            edges_added++;
                        }
                    } catch (const std::out_of_range&) {
                        // Neighbor not found in embedding data, but still add edge with placeholder weight
                        graph.add_edge(vector_id, neighbor_id, 0.0);  // Placeholder weight
                        edges_added++;
                    }
                }
            }
        } catch (const std::out_of_range&) {
            // Vector not found, skip
        }
    }
    
    std::cout << "Built graph with " << graph.size() << " nodes, " 
              << valid_vectors << " valid vectors, " << edges_added << " edges" << std::endl;
    return graph;
}

// load_cagra_data_direct is now defined in simjoin_common.h

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// Multi-cluster helper functions
// ----------------------------------------------------------------------

// Load centroids from file
std::vector<std::vector<float>> load_centroids(const std::string& filename) {
    std::vector<std::vector<float>> centroids;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open centroids file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<float> centroid;
        std::istringstream iss(line);
        float value;
        while (iss >> value) {
            centroid.push_back(value);
        }
        if (!centroid.empty()) {
            centroids.push_back(centroid);
        }
    }
    
    return centroids;
}

// Load centroid distances from file
std::vector<std::vector<double>> load_centroid_distances(const std::string& filename) {
    std::vector<std::vector<double>> distances;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open centroid distances file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<double> cluster_distances;
        std::istringstream iss(line);
        double value;
        while (iss >> value) {
            cluster_distances.push_back(value);
        }
        if (!cluster_distances.empty()) {
            distances.push_back(cluster_distances);
        }
    }
    
    return distances;
}

// Apply triangle inequality pruning
// Returns: (can_prune, is_match_pruned (within epsilon), distance_estimate)
std::tuple<bool, bool, double> apply_triangle_inequality_pruning(
    const std::vector<float>& query_vec, int data_vector_id, double epsilon,
    const std::vector<std::vector<float>>& clusters2_centroids,
    const std::vector<std::vector<double>>& clusters2_centroid_distances,
    const std::unordered_map<int, std::pair<int, int>>& global_to_clusters2_mapping,
    const std::unordered_map<int, std::vector<float>>& global_vectors,
    std::unordered_map<int, double>& query_to_centroid_distances,
    int query_id = -1, const std::string& call_site = "") {  // Add call_site parameter for debugging
    
    auto [data_cluster_id, pos_in_cluster] = global_to_clusters2_mapping.at(data_vector_id);
    
    // Get distance from data vector to its centroid (precomputed)
    double vc_distance = clusters2_centroid_distances[data_cluster_id][pos_in_cluster];
    
    // Get or compute distance from query to centroid
    double xc_distance;
    
    if (query_to_centroid_distances.count(data_cluster_id) > 0) {
        xc_distance = query_to_centroid_distances[data_cluster_id];
    } else {
        // Compute distance from query to centroid
        const auto& centroid = clusters2_centroids[data_cluster_id];
        
        xc_distance = l2_distance(query_vec, centroid);
        query_to_centroid_distances[data_cluster_id] = xc_distance;  // Cache it
    }

    bool debug = false;
    
    if (true) {
        // Apply triangle inequality rules
        // Triangle inequality: |dist(X, C) - dist(Y, C)| ≤ dist(X, Y) ≤ dist(X, C) + dist(Y, C)
        
        // First, compute the actual distance between query vector and data vector for verification
        // global_vectors is an unordered_map<int, vector<float>> where key is global_vector_id

        if (xc_distance + vc_distance <= epsilon) {
            // Rule 1: dist(X, C) + dist(Y, C) ≤ epsilon => dist(X, Y) ≤ epsilon (guaranteed match)
            double pruned_distance = xc_distance + vc_distance;
            
            if (debug) {
                const auto& data_vector = global_vectors.at(data_vector_id);
                double actual_distance = l2_distance(query_vec, data_vector);
                // Assert that pruning is correct: actual distance should be <= epsilon
                if (actual_distance > epsilon) {
                    std::cerr << "ERROR: Triangle inequality pruning assertion failed! "
                            << "Pruned as match but actual distance " << actual_distance 
                            << " > epsilon " << epsilon << std::endl;
                    std::cerr << "Call site: " << call_site << std::endl;
                    std::cerr << "Query ID: " << query_id 
                            << ", Data Global ID: " << data_vector_id
                            << ", Data Cluster ID: " << data_cluster_id
                            << ", Data Local ID: " << pos_in_cluster << std::endl;
                    std::cerr << "xc_distance=" << xc_distance << ", vc_distance=" << vc_distance 
                            << ", pruned_distance=" << pruned_distance << std::endl;
                }
                // Also assert that pruned distance >= actual distance (upper bound)
                if (pruned_distance < actual_distance) {
                    std::cerr << "ERROR: Triangle inequality upper bound violated! "
                            << "Pruned distance " << pruned_distance 
                            << " < actual distance " << actual_distance << std::endl;
                    std::cerr << "Call site: " << call_site << std::endl;
                    std::cerr << "Query ID: " << query_id 
                            << ", Data Global ID: " << data_vector_id
                            << ", Data Cluster ID: " << data_cluster_id
                            << ", Data Local ID: " << pos_in_cluster << std::endl;
                    std::cerr << "xc_distance=" << xc_distance << ", vc_distance=" << vc_distance 
                            << ", pruned_distance=" << pruned_distance << std::endl;
                }
            }
            
            return {true, true, pruned_distance};  // Can prune, is a match
        }
        else if (std::abs(xc_distance - vc_distance) > epsilon) {
            // Rule 2: |dist(X, C) - dist(Y, C)| > epsilon => dist(X, Y) ≥ |dist(X, C) - dist(Y, C)| > epsilon
            // This means dist(X, Y) > epsilon, so it's guaranteed NOT a match
            double pruned_distance = std::abs(xc_distance - vc_distance);
            
            if (debug) {
                const auto& data_vector = global_vectors.at(data_vector_id);
                double actual_distance = l2_distance(query_vec, data_vector);
                // Assert that pruning is correct: actual distance should be > epsilon
                if (actual_distance <= epsilon) {
                    std::cerr << "ERROR: Triangle inequality pruning assertion failed! "
                            << "Pruned as non-match but actual distance " << actual_distance 
                            << " <= epsilon " << epsilon << std::endl;
                    std::cerr << "Call site: " << call_site << std::endl;
                    std::cerr << "Query ID: " << query_id 
                            << ", Data Global ID: " << data_vector_id
                            << ", Data Cluster ID: " << data_cluster_id
                            << ", Data Local ID: " << pos_in_cluster << std::endl;
                    std::cerr << "xc_distance=" << xc_distance << ", vc_distance=" << vc_distance 
                            << ", pruned_distance=" << pruned_distance << std::endl;
                }
                // Also assert that pruned distance <= actual distance (lower bound)
                if (pruned_distance > actual_distance) {
                    std::cerr << "ERROR: Triangle inequality lower bound violated! "
                            << "Pruned distance " << pruned_distance 
                            << " > actual distance " << actual_distance << std::endl;
                    std::cerr << "Call site: " << call_site << std::endl;
                    std::cerr << "Query ID: " << query_id 
                            << ", Data Global ID: " << data_vector_id
                            << ", Data Cluster ID: " << data_cluster_id
                            << ", Data Local ID: " << pos_in_cluster << std::endl;
                    std::cerr << "xc_distance=" << xc_distance << ", vc_distance=" << vc_distance 
                            << ", pruned_distance=" << pruned_distance << std::endl;
                    std::cerr << "Triangle inequality: |" << xc_distance << " - " << vc_distance 
                            << "| = " << pruned_distance << " should be <= " << actual_distance << std::endl;
                }
            }
            
            return {true, false, pruned_distance};  // Can prune, not a match
        }
    }
        
    // Rule 3: Need to compute actual distance
    return {false, false, 0.0};
}


    /**
    * Parse clusters.txt file to get cluster assignments
 * Format: each line i contains space-separated vector IDs in cluster i
 */
std::vector<std::vector<int>> parse_clusters_txt(const std::string& clusters_file) {
    std::cout << "Parsing clusters.txt file: " << clusters_file << std::endl;
    
    // Check if clusters.txt exists
    if (!std::filesystem::exists(clusters_file)) {
        throw std::runtime_error("clusters.txt file not found: " + clusters_file);
    }
    
    std::vector<std::vector<int>> clusters;
    std::ifstream file(clusters_file);
    
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open clusters.txt file: " + clusters_file);
    }
    
    std::string line;
    int cluster_id = 0;
    
    while (std::getline(file, line)) {
        std::vector<int> vector_ids;
        std::istringstream iss(line);
        int vector_id;
        
        // Parse space-separated vector IDs on this line
        while (iss >> vector_id) {
            vector_ids.push_back(vector_id);
        }
        
        // Add this cluster (even if empty)
        clusters.push_back(vector_ids);
        
        if (!vector_ids.empty()) {
            std::cout << "Cluster " << cluster_id << ": " << vector_ids.size() << " vectors" << std::endl;
        } else {
            std::cout << "Cluster " << cluster_id << ": 0 vectors (empty)" << std::endl;
        }
        
        cluster_id++;
    }
    
    file.close();
    std::cout << "Parsed " << clusters.size() << " clusters from " << clusters_file << std::endl;
    return clusters;
}

/**
 * Load cluster data (only cluster_X.nsg, no raw vectors)
 * Returns graph and entry point
 */
std::pair<Graph, int> load_cluster_nsg_only(
    const std::string& cluster_dir, int cluster_id) {
    
    std::string cluster_nsg = cluster_dir + "/cluster_" + std::to_string(cluster_id) + ".nsg";
    
    std::cout << "  Loading cluster " << cluster_id << " NSG index from:" << std::endl;
    std::cout << "    NSG: " << cluster_nsg << std::endl;
    
    // Check if NSG file exists
    if (!std::filesystem::exists(cluster_nsg)) {
        throw std::runtime_error("Cluster NSG file not found: " + cluster_nsg);
    }
    
    // Load cluster NSG index only (no raw vectors)
    auto [cluster_vectors, cluster_neighbors, cluster_entry_point] = load_nsg_data_direct(cluster_nsg);
    
    // Build graph directly from neighbors (CAGRA-only mode)
    Graph cluster_graph;
    int valid_vectors = 0;
    int edges_added = 0;
    
    for (const auto& [vector_id, neighbors] : cluster_neighbors) {
        valid_vectors++;
        for (int neighbor_id : neighbors) {
            // Add edge without computing distance (SimJoin computes distances on-the-fly)
            cluster_graph.add_edge(vector_id, neighbor_id, 0.0);  // Placeholder weight
            edges_added++;
        }
    }
    
    std::cout << "Built graph with " << cluster_graph.size() << " nodes, " 
              << valid_vectors << " valid vectors, " << edges_added << " edges" << std::endl;
    
    std::cout << "  Cluster " << cluster_id << " NSG loaded: " << cluster_graph.size() << " nodes" << std::endl;
    std::cout << "  Cluster " << cluster_id << " entry point: " << cluster_entry_point << std::endl;
    
    return {cluster_graph, cluster_entry_point};
}

std::pair<std::unordered_map<int, std::vector<float>>, Graph> load_cluster_data(
    const std::string& cluster_dir, int cluster_id, int dimension, int cluster_size) {
    
    assert(false);
    
    std::string cluster_fvecs = cluster_dir + "/cluster_" + std::to_string(cluster_id) + ".fvecs";
    std::string cluster_cagra = cluster_dir + "/cluster_" + std::to_string(cluster_id) + ".nsg.cagra";
    
    std::cout << "  Loading cluster " << cluster_id << " from:" << std::endl;
    std::cout << "    Fvecs: " << cluster_fvecs << std::endl;
    std::cout << "    CAGRA: " << cluster_cagra << std::endl;
    
    // Check if files exist
    if (!std::filesystem::exists(cluster_fvecs)) {
        throw std::runtime_error("Cluster fvecs file not found: " + cluster_fvecs);
    }
    
    if (!std::filesystem::exists(cluster_cagra)) {
        throw std::runtime_error("Cluster CAGRA file not found: " + cluster_cagra);
    }
    
    // Load cluster vectors and graph
    auto [cluster_vectors, cluster_neighbors] = load_cagra_data_direct(cluster_cagra);
    
    // Load actual embedding vectors from fvecs file using actual cluster size
    auto cluster_embedding_vectors = load_embedding_vectors(cluster_fvecs, cluster_size, dimension);
    
    // Update cluster vectors with embedding data
    for (auto& [vid, vec] : cluster_vectors) {
        try {
            vec = cluster_embedding_vectors.at(vid);
        } catch (const std::out_of_range&) {
            // Vector not found, keep empty
        }
    }
    
    // Build graph from neighbors
    Graph cluster_graph = build_graph_from_neighbors_simple(cluster_neighbors, cluster_vectors);
    
    std::cout << "  Cluster " << cluster_id << " loaded: " << cluster_vectors.size() 
              << " vectors, " << cluster_graph.size() << " nodes" << std::endl;
    
    return std::make_pair(cluster_vectors, cluster_graph);
}

// ----------------------------------------------------------------------
// NEW: Loader for NSG with OOD flags
// ----------------------------------------------------------------------

std::tuple<std::unordered_map<int, std::vector<float>>, std::unordered_map<int, std::vector<int>>, int> 
load_nsg_with_ood(const std::string& filename) {
    std::cout << "Loading NSG with interleaved OOD flags from: " << filename << "..." << std::endl;
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Error opening NSG file: " + filename);
    }

    unsigned width;
    unsigned ep;
    in.read((char *)&width, sizeof(unsigned));
    in.read((char *)&ep, sizeof(unsigned));

    std::unordered_map<int, std::vector<float>> vectors; // Empty, populated later by embedding loader
    std::unordered_map<int, std::vector<int>> neighbors;
    g_is_node_ood.clear();

    int id = 0;
    while (true) {
        unsigned k;
        in.read((char *)&k, sizeof(unsigned));
        
        // Check for EOF immediately after attempting to read the degree
        if (in.eof()) break;

        std::vector<unsigned> tmp(k);
        if (k > 0) {
            in.read((char *)tmp.data(), k * sizeof(unsigned));
        }
        
        // Convert unsigned neighbors to int for compatibility
        std::vector<int> nbrs(tmp.begin(), tmp.end());
        neighbors[id] = nbrs;

        // Read the interleaved OOD byte immediately after the neighbors
        unsigned char is_id_byte;
        in.read((char *)&is_id_byte, sizeof(unsigned char));
        
        // Byte 1 = ID (true), Byte 0 = OOD (false)
        // We want to store "is_ood", so if byte != 1 (i.e. 0), it is OOD
        bool is_ood = (is_id_byte != 1);
        g_is_node_ood.push_back(is_ood);

        id++;
    }
    
    std::cout << "Loaded " << id << " nodes. OOD Count: " 
              << std::count(g_is_node_ood.begin(), g_is_node_ood.end(), true) << std::endl;

    return {vectors, neighbors, (int)ep};
}

std::tuple<std::unordered_map<int, std::vector<float>>, std::unordered_map<int, std::vector<int>>, int> 
load_nsg_ood_data_direct(const std::string& nsg_file) {
    std::cout << "Loading NSG file with interleaved OOD flags: " << nsg_file << std::endl;
    
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
    g_is_node_ood.clear();
    
    uint32_t vector_count = 0;
    while (true) {
        // Try to read k (degree)
        uint32_t k;
        file.read(reinterpret_cast<char*>(&k), 4);
        
        // Check for EOF immediately after trying to read the degree
        if (file.gcount() != 4) {
            break; // End of file
        }
        
        // Read k neighbor indices
        std::vector<int> neighbors;
        neighbors.reserve(k);
        
        // Reading neighbors in bulk is faster than one by one
        if (k > 0) {
            std::vector<uint32_t> temp_neighbors(k);
            file.read(reinterpret_cast<char*>(temp_neighbors.data()), k * 4);
            
            if (file.gcount() != k * 4) {
                std::cout << "Error: Unexpected EOF reading neighbors for node " << vector_count << std::endl;
                break; // Incomplete data
            }
            
            // Cast uint32 to int
            for (uint32_t n : temp_neighbors) {
                neighbors.push_back(static_cast<int>(n));
            }
        }
        
        // Read OOD byte (interleaved)
        unsigned char is_id_byte;
        file.read(reinterpret_cast<char*>(&is_id_byte), 1);
        if (file.gcount() != 1) {
            std::cout << "Error: Unexpected EOF reading OOD byte for node " << vector_count << std::endl;
            break; 
        }

        // Logic: Byte 1 = ID (is_ood=false), Byte 0 = OOD (is_ood=true)
        // We push back result: is_ood
        g_is_node_ood.push_back(is_id_byte != 1);

        // Initialize empty vector (will be filled by embedding files later in main)
        vectors[vector_count] = std::vector<float>();
        graph_edges[vector_count] = neighbors;
        vector_count++;
        
        if (vector_count % 100000 == 0) {
            std::cout << "  Processed " << vector_count << " vectors..." << std::endl;
        }
    }
    
    std::cout << "  Total vectors: " << vector_count << std::endl;
    
    // Calculate average degree for logging
    double avg_degree = (vector_count > 0) ? 
        (double)std::accumulate(graph_edges.begin(), graph_edges.end(), 0.0, 
            [](double sum, const auto& pair) { return sum + pair.second.size(); }) / vector_count 
        : 0.0;
        
    std::cout << "  Average degree: " << avg_degree << std::endl;
    
    size_t ood_count = std::count(g_is_node_ood.begin(), g_is_node_ood.end(), true);
    std::cout << "  OOD Nodes found: " << ood_count << " (" 
              << (vector_count > 0 ? (100.0 * ood_count / vector_count) : 0.0) << "%)" << std::endl;
    
    return {vectors, graph_edges, static_cast<int>(ep_)};
}

// ----------------------------------------------------------------------
// Main execution
// ----------------------------------------------------------------------

int main(int argc, char* argv[]) {
    if (argc < 31) {
        std::cout << "Usage: " << argv[0] << " <result_dir> <epsilon> <w_queue> <num_threads> <supplier_nsg> <part_nsg> <supplier_embedding> <part_embedding> <prefix> <Q> <dimension> <force_kappa_y0> <early_terminate> <enable_top1_detection> <enable_closest_fallback> <enable_query_to_data_edges> <k_top_data_points> <collect_bfs_data> <break_before_bfs> <seed_offset> <sort_jj_by_distance> <enable_seed_offset_filtering> <cache_closest_only> <max_bfs_out_of_range_tolerance> <adaptive_bfs_threshold_factor> <one_hop_data_only> [clusters1.txt] [clusters2.txt]" << std::endl;
        std::cout << "  result_dir: result directory name (or 'auto' to generate automatically)" << std::endl;
        std::cout << "  epsilon: similarity threshold" << std::endl;
        std::cout << "  w_queue: window queue size" << std::endl;
        std::cout << "  num_threads: number of threads" << std::endl;
        std::cout << "  supplier_nsg: supplier NSG graph file" << std::endl;
        std::cout << "  part_nsg: part NSG graph file" << std::endl;
        std::cout << "  supplier_embedding: supplier embedding file" << std::endl;
        std::cout << "  part_embedding: part embedding file" << std::endl;
        std::cout << "  prefix: prefix for result directory" << std::endl;
        std::cout << "  Q: Number of query vectors to process (first Q vectors from supplier embedding)" << std::endl;
        std::cout << "  dimension: Dimension of the embedding vectors" << std::endl;
        std::cout << "  force_kappa_y0: Force kappa_i = y0 for all nodes (0=false, 1=true)" << std::endl;
        std::cout << "  early_terminate: Early termination when any match found (0=false, 1=true)" << std::endl;
        std::cout << "  enable_top1_detection: Enable top-1 nearest node detection (0=false, 1=true)" << std::endl;
        std::cout << "  enable_closest_fallback: Enable closest result fallback when J_j is empty (0=false, 1=true)" << std::endl;
        std::cout << "  enable_query_to_data_edges: Enable adding join results as edges in data index (0=false, 1=true)" << std::endl;
        std::cout << "  k_top_data_points: When query-to-data edges enabled, connect top-k data points with edges (0=all, >0=top-k)" << std::endl;
        std::cout << "  collect_bfs_data: Collect BFS seeds and visited nodes (0=false, 1=true)" << std::endl;
        std::cout << "  break_before_bfs: Break before BFS (0=false, 1=true)" << std::endl;
        std::cout << "  seed_offset: Offset to add to query index x_i for seed selection (0=use default, >0=add offset)" << std::endl;
        std::cout << "  sort_jj_by_distance: Sort J_j by distance to x_j (0=false, 1=true)" << std::endl;
        std::cout << "  enable_seed_offset_filtering: Enable filtering of nodes >= seed_offset in J_j (0=false, 1=true)" << std::endl;
        std::cout << "  cache_closest_only: Cache closest node only when enable_closest_fallback is enabled (0=false, 1=true)" << std::endl;
        std::cout << "  max_bfs_out_of_range_tolerance: Max BFS out-of-range tolerance - push out-of-range points to BFS queue and terminate after N consecutive out-of-range points (0=disabled, >0=threshold)" << std::endl;
        std::cout << "  adaptive_bfs_threshold_factor: Adaptive BFS threshold factor - dynamically adjust BFS threshold based on distance std deviation (0.0=disabled, >0.0=epsilon + factor/std_dev)" << std::endl;
        std::cout << "  one_hop_data_only: Enable data seed fallback mode (0=false, 1=true)" << std::endl;
        std::cout << "  topK: Number of top data points to connect with edges when query-to-data edges enabled (0=all, >0=top-k)" << std::endl;
        std::cout << "  patience: Number of consecutive out-of-range points to tolerate during BFS before termination (0=disabled, >0=threshold)" << std::endl;
        std::cout << "  ood_file_name: File containing OOD flags for supplier NSG nodes" << std::endl;
        std::cout << "  number_cached: Number of cached closest nodes when cache_closest_only is enabled" << std::endl;
        std::cout << "  clusters1.txt: Optional clusters file for multi-cluster mode" << std::endl;
        std::cout << "  clusters2.txt: Optional clusters file for triangle inequality pruning" << std::endl;
        std::cout << "    If provided, will process each cluster separately and combine results" << std::endl;
        return 1;
    }
    
    std::string result_dir_arg = argv[1];
    double epsilon = std::stod(argv[2]);
    int w_queue = std::stoi(argv[3]);
    int num_threads = std::stoi(argv[4]);
    std::string supplier_nsg = argv[5];
    std::string part_nsg = argv[6];
    std::string supplier_embedding = argv[7];
    std::string part_embedding = argv[8];
    std::string prefix = argv[9];
    int Q = std::stoi(argv[10]);
    int dimension = std::stoi(argv[11]);
    bool force_kappa_y0 = (std::stoi(argv[12]) != 0); // don't use MST-based parent-child relationships, always use y0
    bool early_terminate = (std::stoi(argv[13]) != 0); // break when any match found during the seed iteration
    bool enable_top1_detection = (std::stoi(argv[14]) != 0); // break when the top-1 distance does not decrease during the greedy search
    bool enable_closest_fallback = (std::stoi(argv[15]) != 0); // return closest result when J_j is empty -- sort work sharing
    bool enable_query_to_data_edges = (std::stoi(argv[16]) != 0); // add join results as query-to-data edges to the data index -- work sharing via edges
    int k_top_data_points = std::stoi(argv[17]); // when query-to-data edges enabled, connect top-k data points with edges (0=all, >0=top-k)
    bool collect_bfs_data = (std::stoi(argv[18]) != 0); // collect BFS seeds and visited nodes
    bool break_before_bfs = (std::stoi(argv[19]) != 0); // break before BFS
    int seed_offset = std::stoi(argv[20]); // offset to add to query index x_i for seed selection
    bool sort_jj_by_distance = (std::stoi(argv[21]) != 0); // sort J_j by distance to x_j
    bool enable_seed_offset_filtering = (std::stoi(argv[22]) != 0); // enable filtering of nodes >= seed_offset in J_j
    bool cache_closest_only = (std::stoi(argv[23]) != 0); // enable tracking of closest node for window filling
    int max_bfs_out_of_range_tolerance = std::stoi(argv[24]); // max BFS out-of-range tolerance (0=disabled)
    double adaptive_bfs_threshold_factor = std::stod(argv[25]); // adaptive BFS threshold factor (0.0=disabled)
    bool one_hop_data_only = (std::stoi(argv[26]) != 0); // enable data seed fallback mode
    int topK = std::stoi(argv[27]);
    int patience = std::stoi(argv[28]);
    std::string ood_file_name = argv[29];
    int number_cached = std::stoi(argv[30]);
    bool merged_index = false;      // wether or not we use the merged index or not 


    std::cout << "TOPK iss " << topK << std::endl;


    std::cout << "topK: " << topK << " patience: " << patience << std::endl;
    
    std::cout << "=== Queue-Based Parallel SimJoin (C++) - CORRECTED ===" << std::endl;
    if (force_kappa_y0) {
        std::cout << "MODE: Force kappa_i = y0 for all nodes (no seed inheritance from parent nodes)" << std::endl;
    } else {
        std::cout << "MODE: Normal MST-based parent-child relationships" << std::endl;
    }
    if (early_terminate) {
        std::cout << "MODE: Early termination enabled - break loop when any match found" << std::endl;
    } else {
        std::cout << "MODE: Early termination disabled - process all seeds" << std::endl;
    }
    if (enable_top1_detection) {
        std::cout << "MODE: Top-1 detection enabled - break when nearest data node found" << std::endl;
    } else {
        std::cout << "MODE: Top-1 detection disabled - continue until convergence" << std::endl;
    }
    if (enable_closest_fallback) {
        std::cout << "MODE: Closest fallback enabled - return closest result when J_j is empty" << std::endl;
    } else {
        std::cout << "MODE: Closest fallback disabled - return empty set when J_j is empty" << std::endl;
    }
    if (enable_query_to_data_edges) {
        std::cout << "MODE: Query-to-data edges enabled - add join results as edges in data index" << std::endl;
    } else {
        std::cout << "MODE: Query-to-data edges disabled - no edges added to data index" << std::endl;
    }
    if (seed_offset > 0) {
        std::cout << "MODE: Seed offset enabled - adding " << seed_offset << " to query index x_i for seed selection" << std::endl;
    } else {
        std::cout << "MODE: Seed offset disabled - using default seed selection" << std::endl;
    }
    if (sort_jj_by_distance) {
        std::cout << "MODE: J_j sorting enabled - sort join results by distance to x_j (smallest to largest)" << std::endl;
    } else {
        std::cout << "MODE: J_j sorting disabled - return join results in arbitrary order" << std::endl;
    }
    if (enable_seed_offset_filtering) {
        std::cout << "MODE: Seed offset filtering enabled - filtering out nodes >= " << seed_offset << " in J_j" << std::endl;
    } else {
        std::cout << "MODE: Seed offset filtering disabled - no filtering in J_j" << std::endl;
    }
    if (cache_closest_only) {
        std::cout << "MODE: Caching closest node only enabled - tracking closest node for window filling" << std::endl;
    } else {
        std::cout << "MODE: Caching closest node only disabled - normal window filling" << std::endl;
    }
    if (max_bfs_out_of_range_tolerance > 0) {
        std::cout << "MODE: BFS out-of-range tolerance enabled - push out-of-range points to BFS queue and terminate after " << max_bfs_out_of_range_tolerance << " consecutive out-of-range points" << std::endl;
    } else {
        std::cout << "MODE: BFS out-of-range tolerance disabled - continue BFS until queue is empty" << std::endl;
    }
    if (adaptive_bfs_threshold_factor > 0.0) {
        std::cout << "MODE: Adaptive BFS threshold enabled - dynamically adjust threshold based on distance std deviation (factor=" << adaptive_bfs_threshold_factor << ")" << std::endl;
    } else {
        std::cout << "MODE: Adaptive BFS threshold disabled - use fixed epsilon threshold" << std::endl;
    }
    if (topK > 0) {
        std::cout << "MODE: When query-to-data edges enabled, connect top-" << topK << " data points with edges" << std::endl;
    } else {
        std::cout << "MODE: When query-to-data edges enabled, connect all data points with edges" << std::endl;
    }
    if (patience > 0) {
        std::cout << "MODE: BFS out-of-range patience enabled - terminate after " << patience << " consecutive out-of-range points" << std::endl;
    } else {
        std::cout << "MODE: BFS out-of-range patience disabled - no patience threshold" << std::endl;
    }
    
    // Set up signal handling for graceful shutdown
    signal(SIGINT, signal_handler);   // Ctrl+C
    signal(SIGTERM, signal_handler);  // Termination signal
    signal(SIGUSR1, signal_handler);  // User-defined signal 1
    
    std::cout << "Profiled SimJoin started. Press Ctrl+C to stop and save partial results." << std::endl;
    
    // Create result directory name for profiling results
    std::string result_dir;
    if (result_dir_arg == "auto") {
        // Generate directory name automatically
        result_dir = "simjoin_result/" + prefix + "_epsilon_" + std::to_string(epsilon) + "_w_" + std::to_string(w_queue) + "_threads_" + std::to_string(num_threads) + "_Q" + std::to_string(Q);
        if (force_kappa_y0) {
            result_dir += "_NOWS"; // no work sharing
        }
        if (early_terminate) {
            result_dir += "_ESS"; // early stop seed
        }
        if (enable_top1_detection) {
            result_dir += "_ESN"; // early stop neighbor
        }
        if (enable_closest_fallback) {
            result_dir += "_SWS"; // soft work sharing
        }
        if (enable_query_to_data_edges) {
            result_dir += "_QD"; // query-to-data edges
            if (k_top_data_points > 0) {
                result_dir += "_Top" + std::to_string(k_top_data_points); // add top-k
            }
        }
        if (collect_bfs_data) {
            result_dir += "_BFSC"; // BFS data collection
        }
        if (break_before_bfs) {
            result_dir += "_BBF"; // break before BFS
        }
        if (seed_offset > 0) {
            result_dir += "_SO" + std::to_string(seed_offset); // seed offset
        }
        // new parameters
        if (sort_jj_by_distance) {
            result_dir += "_SJ"; // sort J_j by distance
        }
        if (enable_seed_offset_filtering) {
            result_dir += "_SOF"; // seed offset filtering
        }
        if (cache_closest_only) {
            result_dir += "_CC"; // closest caching only
        }
        if (max_bfs_out_of_range_tolerance > 0) {
            result_dir += "_JUMP" + std::to_string(max_bfs_out_of_range_tolerance); // BFS out-of-range tolerance
        }
        if (adaptive_bfs_threshold_factor > 0.0) {
            result_dir += "_ADAPT" + std::to_string(adaptive_bfs_threshold_factor); // Adaptive threshold (multiply by 100 for integer suffix)
        }
        if (one_hop_data_only) {
            result_dir += "_OH";
        }
        if (topK > 0) {
            result_dir += "_Top" + std::to_string(topK);
        }
        if (patience > 0) {
            result_dir += "_Pat" + std::to_string(patience);
        }
        if (number_cached > 0) {
            result_dir += "_NC" + std::to_string(number_cached);
        }
    } else {
        // Use provided directory name
        result_dir = result_dir_arg;
    }
    
    try {
        // Load supplier data from NSG file
        auto [supplier_vectors, supplier_neighbors, supplier_entry_point] = load_nsg_data_direct(supplier_nsg);
        // Load part data from NSG file (only needed for single-cluster mode)
        std::unordered_map<int, std::vector<float>> part_vectors;
        std::unordered_map<int, std::vector<int>> part_neighbors;
        Graph Gy;
        int part_entry_point = -1;  // Initialize entry point
        
        // Load actual embedding vectors for supplier data (will be loaded after cluster check)
        
        // Debug: Check vector ID ranges
        int min_supplier_cagra_id = INT_MAX, max_supplier_cagra_id = -1;
        for (const auto& [vid, _] : supplier_vectors) {
            min_supplier_cagra_id = std::min(min_supplier_cagra_id, vid);
            max_supplier_cagra_id = std::max(max_supplier_cagra_id, vid);
        }
        
        std::cout << "Supplier CAGRA vector ID range: " << min_supplier_cagra_id << " to " << max_supplier_cagra_id << std::endl;
        
        // Build supplier graph (will be done after supplier loading)


        if (supplier_embedding.find("all") != std::string::npos &&
            part_embedding.find("all") != std::string::npos) {
            merged_index = true;
        } else {
            merged_index = false;
        }

        std::unordered_map<int, bool> ood_flags;
        {
            std::ifstream ood_file(ood_file_name);
            if (!ood_file.is_open()) {
                throw std::runtime_error("Cannot open OOD file: " + ood_file_name);
            }

            std::string line;
            size_t count = 0;
            int current_id = 0; // Tracks the ID implicitly by line number

            while (std::getline(ood_file, line)) {
                std::istringstream iss(line);
                int flag_val;

                // Read the 0 or 1 value from the line
                if (iss >> flag_val) {
                    // Only store if the flag is 1 (true)
                    if (flag_val == 1) {
                        ood_flags[current_id] = true;
                        ++count;
                    }
                }
                
                current_id++;
            }
            std::cout << "Loaded " << count << " OOD entries from " << ood_file_name 
                    << " (Scanned " << current_id << " lines)" << std::endl;
        }
        
        // Determine cluster configuration based on command line arguments
        std::string clusters1_txt = "";
        std::string clusters2_txt = "";
        
        if (argc > 31) {
            clusters1_txt = argv[31];
        }
        if (argc > 32) {
            clusters2_txt = argv[32];
        }
        
        // Create directory if it doesn't exist
        std::filesystem::create_directories(result_dir);
        
        // Create filenames for results
        std::string mst_filename = result_dir + "/mst_data.txt";
        
        // Check if clusters.txt is provided for multi-cluster mode
        bool use_clusters = false;
        std::vector<std::vector<int>> cluster_assignments;
        std::vector<std::unordered_map<int, std::vector<float>>> cluster_vectors_list;
        std::vector<Graph> cluster_graphs_list;
        std::vector<std::unordered_map<int, int>> local_to_global_mapping;
        std::vector<int> cluster_entry_points;
        std::vector<std::unordered_map<int, std::vector<float>>> cluster_local_vectors_list;
        
        // Triangle inequality pruning data structures
        bool use_triangle_inequality = false;
        std::vector<std::vector<int>> clusters2_assignments;
        std::vector<std::vector<float>> clusters2_centroids;
        std::vector<std::vector<double>> clusters2_centroid_distances;
        std::unordered_map<int, std::pair<int, int>> global_to_clusters2_mapping;  // global_id -> (cluster_id, position_in_cluster)
        std::unordered_map<int, double> query_to_centroid_distances;  // Cache for dist(X, C) - key is cluster_id
        
        int total_matches = 0;  // Declare here for use in results section

        if (!clusters1_txt.empty()) {
            result_dir += "_C1";
        }

        if (!clusters2_txt.empty()) {
            result_dir += "_C2";
        }
        
        // Case 1: No clusters specified - use global index, no pruning
        if (clusters1_txt.empty() && clusters2_txt.empty()) {
            std::cout << "Mode: Global index, no clustering, no triangle inequality pruning" << std::endl;
            use_clusters = false;
            use_triangle_inequality = false;
            std::cout << "DEBUG: use_triangle_inequality = " << use_triangle_inequality << std::endl;
        }
        // Case 2: Only clusters1 specified - use per-cluster index, no pruning
        else if (!clusters1_txt.empty() && clusters2_txt.empty()) {
            std::cout << "Mode: Per-cluster index, no triangle inequality pruning" << std::endl;
            use_clusters = true;
            use_triangle_inequality = false;
            
            cluster_assignments = parse_clusters_txt(clusters1_txt);
            std::cout << "Found " << cluster_assignments.size() << " clusters" << std::endl;
        }
        // Case 3: Only clusters2 specified - use global index with pruning
        else if (clusters1_txt.empty() && !clusters2_txt.empty()) {
            std::cout << "Mode: Global index with triangle inequality pruning" << std::endl;
            use_clusters = false;
            use_triangle_inequality = true;
            
            clusters2_assignments = parse_clusters_txt(clusters2_txt);
            std::cout << "Found " << clusters2_assignments.size() << " clusters2 for triangle inequality pruning" << std::endl;
            
            // Load centroids and centroid distances for clusters2
            std::string clusters2_dir = std::filesystem::path(clusters2_txt).parent_path().string();
            if (clusters2_dir.empty()) clusters2_dir = ".";
            
            // Load centroids
            std::string centroids_file = clusters2_dir + "/centroids.txt";
            if (std::filesystem::exists(centroids_file)) {
                clusters2_centroids = load_centroids(centroids_file);
                std::cout << "Loaded " << clusters2_centroids.size() << " centroids for triangle inequality pruning" << std::endl;
            } else {
                std::cout << "WARNING: Centroids file not found: " << centroids_file << std::endl;
                use_triangle_inequality = false;
            }
            
            // Load centroid distances
            std::string centroid_distances_file = clusters2_dir + "/centroid_distances.txt";
            if (std::filesystem::exists(centroid_distances_file)) {
                clusters2_centroid_distances = load_centroid_distances(centroid_distances_file);
                std::cout << "Loaded centroid distances for triangle inequality pruning" << std::endl;
            } else {
                std::cout << "WARNING: Centroid distances file not found: " << centroid_distances_file << std::endl;
                use_triangle_inequality = false;
            }
            
            // Build global to clusters2 mapping
            for (size_t cluster_id = 0; cluster_id < clusters2_assignments.size(); cluster_id++) {
                for (size_t pos = 0; pos < clusters2_assignments[cluster_id].size(); pos++) {
                    int global_id = clusters2_assignments[cluster_id][pos];
                    global_to_clusters2_mapping[global_id] = {static_cast<int>(cluster_id), static_cast<int>(pos)};
                }
            }
            std::cout << "Built global to clusters2 mapping for " << global_to_clusters2_mapping.size() << " vectors" << std::endl;
        }
        // Case 4: Both clusters1 and clusters2 specified - use per-cluster index with pruning
        else if (!clusters1_txt.empty() && !clusters2_txt.empty()) {
            std::cout << "Mode: Per-cluster index with triangle inequality pruning" << std::endl;
            use_clusters = true;
            use_triangle_inequality = true;
            
            // Load clusters1 for main processing
            cluster_assignments = parse_clusters_txt(clusters1_txt);
            std::cout << "Found " << cluster_assignments.size() << " clusters" << std::endl;
            
            // Load clusters2 for triangle inequality pruning
            clusters2_assignments = parse_clusters_txt(clusters2_txt);
            std::cout << "Found " << clusters2_assignments.size() << " clusters2 for triangle inequality pruning" << std::endl;
            
            // Load centroids and centroid distances for clusters2
            std::string clusters2_dir = std::filesystem::path(clusters2_txt).parent_path().string();
            if (clusters2_dir.empty()) clusters2_dir = ".";
            
            // Load centroids
            std::string centroids_file = clusters2_dir + "/centroids.txt";
            if (std::filesystem::exists(centroids_file)) {
                clusters2_centroids = load_centroids(centroids_file);
                std::cout << "Loaded " << clusters2_centroids.size() << " centroids for triangle inequality pruning" << std::endl;
            } else {
                std::cout << "WARNING: Centroids file not found: " << centroids_file << std::endl;
                use_triangle_inequality = false;
            }
            
            // Load centroid distances
            std::string centroid_distances_file = clusters2_dir + "/centroid_distances.txt";
            if (std::filesystem::exists(centroid_distances_file)) {
                clusters2_centroid_distances = load_centroid_distances(centroid_distances_file);
                std::cout << "Loaded centroid distances for triangle inequality pruning" << std::endl;
            } else {
                std::cout << "WARNING: Centroid distances file not found: " << centroid_distances_file << std::endl;
                use_triangle_inequality = false;
            }
            
            // Build global to clusters2 mapping
            for (size_t cluster_id = 0; cluster_id < clusters2_assignments.size(); cluster_id++) {
                for (size_t pos = 0; pos < clusters2_assignments[cluster_id].size(); pos++) {
                    int global_id = clusters2_assignments[cluster_id][pos];
                    global_to_clusters2_mapping[global_id] = {static_cast<int>(cluster_id), static_cast<int>(pos)};
                }
            }
            std::cout << "Built global to clusters2 mapping for " << global_to_clusters2_mapping.size() << " vectors" << std::endl;
        }
        
        // Load global part vectors (needed for all modes)
        std::cout << "Loading global part vectors from: " << part_embedding << std::endl;
        auto global_part_vectors = load_embedding_vectors(part_embedding, -1, dimension);  // -1 means load all
        std::cout << "Loaded " << global_part_vectors.size() << " global part vectors" << std::endl;
        
        // Calculate data offset for query-to-data edge mapping (for all modes)
        int data_offset = global_part_vectors.size();  // Query IDs will be mapped to data space by adding this offset
        if (enable_query_to_data_edges) {
            std::cout << "  Data offset for query-to-data edge mapping: " << data_offset << std::endl;
        }
        
        // Store global vectors for use in similarity calculations
        cluster_vectors_list.push_back(global_part_vectors);
        
        // Load per-cluster CAGRA indices only (no raw vectors) - only for cluster modes
        if (use_clusters) {
            std::string cluster_dir = std::filesystem::path(clusters1_txt).parent_path().string();
            if (cluster_dir.empty()) cluster_dir = ".";
            
            for (size_t cluster_id = 0; cluster_id < cluster_assignments.size(); cluster_id++) {
                std::cout << "Loading NSG index for cluster " << cluster_id << "..." << std::endl;
                auto [cluster_graph, cluster_entry_point] = load_cluster_nsg_only(cluster_dir, static_cast<int>(cluster_id));
                cluster_graphs_list.push_back(cluster_graph);
                cluster_entry_points.push_back(cluster_entry_point);
            }
            
            // Create local-to-global ID mapping for each cluster
            for (size_t cluster_id = 0; cluster_id < cluster_assignments.size(); cluster_id++) {
                std::unordered_map<int, int> mapping;
                for (size_t local_id = 0; local_id < cluster_assignments[cluster_id].size(); local_id++) {
                    int global_id = cluster_assignments[cluster_id][local_id];
                    mapping[local_id] = global_id;
                }
                local_to_global_mapping.push_back(mapping);
            }
        } else {
            // Global mode: use single global NSG index
            std::cout << "Loading global NSG index..." << std::endl;
            auto [part_vectors_temp, part_neighbors_temp, part_entry_point_temp] = load_nsg_data_direct(part_nsg);
            // auto [part_vectors_dummy, part_neighbors_dummy, part_entry_point_dummy] = load_nsg_ood_data_direct("./laion/laion_all_ood.nsg");
            part_vectors = part_vectors_temp;
            part_neighbors = part_neighbors_temp;
            part_entry_point = part_entry_point_temp;

            // Create a single cluster with all vectors
            std::vector<int> all_global_ids;
            for (const auto& [vid, _] : global_part_vectors) {
                all_global_ids.push_back(vid);
            }
            cluster_assignments.push_back(all_global_ids);
            
            // Create local-to-global mapping for global mode
            std::unordered_map<int, int> global_mapping;
            for (size_t local_id = 0; local_id < all_global_ids.size(); local_id++) {
                int global_id = all_global_ids[local_id];
                global_mapping[local_id] = global_id;
            }
            local_to_global_mapping.push_back(global_mapping);

            // Use actual part vector count from NSG file instead of hardcoded value
            const int PART_DIM = dimension;
            const int PART_COUNT = part_vectors_temp.size();  // Infer from actual loaded data
            std::cout << "  Inferred part vector count from NSG file: " << PART_COUNT << std::endl;
            
            auto part_embedding_vectors = load_embedding_vectors(part_embedding, PART_COUNT, PART_DIM);
            
            // Update part vectors with embedding data
            int part_vectors_with_data = 0;
            for (auto& [vid, vec] : part_vectors) {
                try {
                    vec = part_embedding_vectors.at(vid);
                    part_vectors_with_data++;
                } catch (const std::out_of_range&) {
                    // Vector not found, keep empty
                }
            }
            std::cout << "Part vectors with embedding data: " << part_vectors_with_data << "/" << part_vectors.size() << std::endl;
            
            // Build part graph
            std::cout << "Building part graph (without distances)..." << std::endl;
            Gy = build_graph_from_neighbors_simple(part_neighbors, part_vectors);

            cluster_graphs_list.push_back(Gy);
            std::cout << "Part graph: " << Gy.size() << " nodes" << std::endl;
        }
        
        // Load actual embedding vectors for supplier data (after cluster check)
        const int SUPPLIER_DIM = dimension;
        const int SUPPLIER_COUNT = -1;     // Always load all vectors for consistency across all modes
        
        auto supplier_embedding_vectors = load_embedding_vectors(supplier_embedding, SUPPLIER_COUNT, SUPPLIER_DIM);
        
        // Debug: Check supplier embedding vector ID range
        int min_supplier_embedding_id = INT_MAX, max_supplier_embedding_id = -1;
        for (const auto& [vid, _] : supplier_embedding_vectors) {
            min_supplier_embedding_id = std::min(min_supplier_embedding_id, vid);
            max_supplier_embedding_id = std::max(max_supplier_embedding_id, vid);
        }
        std::cout << "Supplier embedding vector ID range: " << min_supplier_embedding_id << " to " << max_supplier_embedding_id << std::endl;
        
        // Update supplier vectors with actual embedding data
        int supplier_vectors_with_data = 0;
        for (auto& [vid, vec] : supplier_vectors) {
            try {
                vec = supplier_embedding_vectors.at(vid);
                supplier_vectors_with_data++;
            } catch (const std::out_of_range&) {
                // Vector not found, keep empty
            }
        }
        
        std::cout << "Supplier vectors with embedding data: " << supplier_vectors_with_data << "/" << supplier_vectors.size() << std::endl;
        
        // Build supplier graph (after supplier vectors are populated with embedding data)
        std::cout << "Building supplier graph (with distances for MST)..." << std::endl;
        Graph Gx = build_graph_from_neighbors_with_distances(supplier_neighbors, supplier_vectors);
        std::cout << "Supplier graph: " << Gx.size() << " nodes" << std::endl;
        
        // Check connectivity of query index after loading
        check_query_connectivity_after_loading(Gx, supplier_vectors);
        
        // Run similarity join with timing
        std::cout << "\nRunning full simjoin with epsilon=" << epsilon << "..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        if (use_clusters) {
            // Multi-cluster mode: process each cluster separately
            std::cout << "Processing " << cluster_assignments.size() << " clusters..." << std::endl;
            
            
            // Clear global variables for this run
            g_cluster_windows.clear();
            g_cluster_processing_times.clear();
            
            // Create per-query query_to_centroid_distances mapping for multi-cluster mode
            std::unordered_map<int, std::unordered_map<int, double>> query_to_centroid_distances_map;
            
            // Initialize query_to_centroid_distances_map with all query IDs
            for (const auto& [query_id, query_vec] : supplier_vectors) {
                if (!query_vec.empty()) {
                    query_to_centroid_distances_map[query_id] = std::unordered_map<int, double>();
                }
            }
            std::cout << "Initialized query_to_centroid_distances_map for " << query_to_centroid_distances_map.size() << " queries" << std::endl;
            
            std::cout << "DEBUG: g_cluster_windows cleared, size: " << g_cluster_windows.size() << std::endl;
            
            for (size_t cluster_id = 0; cluster_id < cluster_assignments.size(); cluster_id++) {
                // Check for shutdown request before processing each cluster
                if (g_shutdown_requested) {
                    std::cout << "Shutdown requested during cluster processing, stopping at cluster " << cluster_id << std::endl;
                    break;
                }
                
                std::cout << "Processing cluster " << cluster_id << "..." << std::endl;
                
                // Create per-cluster output filename
                std::string cluster_output_filename = result_dir + "/cluster_" + std::to_string(cluster_id) + "_join_output.txt";
                std::cout << "  Using output filename: " << cluster_output_filename << std::endl;
                
                // Create local-to-global vector mapping for this cluster
                std::unordered_map<int, std::vector<float>> cluster_local_vectors;
                for (const auto& [local_id, global_id] : local_to_global_mapping[cluster_id]) {
                    if (cluster_vectors_list[0].count(global_id) > 0) {
                        cluster_local_vectors[local_id] = cluster_vectors_list[0][global_id];
                    }
                }
                
                // Store local vectors for this cluster for later use in result processing
                cluster_local_vectors_list.push_back(cluster_local_vectors);
                
                std::cout << "DEBUG: Cluster " << cluster_id << " has " << cluster_local_vectors.size() << " local vectors mapped from global vectors" << std::endl;
                
                // Call sim_join for this cluster with local vectors and CAGRA index
                std::cout << "DEBUG: Calling sim_join for cluster " << cluster_id << " with " << supplier_vectors.size() << " supplier vectors and " << cluster_local_vectors.size() << " local cluster vectors" << std::endl;
                sim_join(supplier_vectors, cluster_local_vectors, Gx, cluster_graphs_list[cluster_id], 
                        epsilon, w_queue, num_threads, mst_filename, false, cluster_output_filename, result_dir, 
                        force_kappa_y0, early_terminate, enable_top1_detection, enable_closest_fallback, static_cast<int>(cluster_id),
                        use_triangle_inequality, 
                        use_triangle_inequality ? &clusters2_centroids : nullptr,
                        use_triangle_inequality ? &clusters2_centroid_distances : nullptr,
                        use_triangle_inequality ? &global_to_clusters2_mapping : nullptr,
                        use_triangle_inequality ? &cluster_vectors_list[0] : nullptr,
                        use_triangle_inequality ? &query_to_centroid_distances : nullptr,
                        &local_to_global_mapping,
                        use_triangle_inequality ? &query_to_centroid_distances_map : nullptr,
                        enable_query_to_data_edges,
                        k_top_data_points,
                        data_offset,
                        collect_bfs_data,
                        break_before_bfs,
                        seed_offset,
                        enable_seed_offset_filtering,
                        sort_jj_by_distance,
                        cache_closest_only, 
                        cluster_entry_points[cluster_id], 
                        max_bfs_out_of_range_tolerance,
                        adaptive_bfs_threshold_factor,
                        one_hop_data_only,
                        topK,
                        patience,
                        ood_flags,
                        number_cached,
                        merged_index
                    ); // no per-query mapping in multi-cluster mode
                // std::cout << "DEBUG: After sim_join, g_cluster_windows size: " << g_cluster_windows.size() << std::endl;
                
                // Check if the file was actually created
                if (std::filesystem::exists(cluster_output_filename)) {
                    std::cout << "  Cluster " << cluster_id << " completed, output written to " << cluster_output_filename << std::endl;
                } else {
                    std::cout << "  WARNING: Cluster " << cluster_id << " output file was NOT created: " << cluster_output_filename << std::endl;
                }
            }
            
            // Now write final output with local to global ID conversion
            std::cout << "Writing final output with local to global ID conversion..." << std::endl;
            std::cout << "DEBUG: g_cluster_windows final size: " << g_cluster_windows.size() << std::endl;
            for (size_t i = 0; i < g_cluster_windows.size(); i++) {
                std::cout << "DEBUG: Cluster " << i << " has " << g_cluster_windows[i].size() << " query windows" << std::endl;
                if (g_cluster_windows[i].size() > 0) {
                    std::cout << "DEBUG: Cluster " << i << " query IDs: ";
                    int count = 0;
                    for (const auto& [query_id, window] : g_cluster_windows[i]) {
                        if (count < 3) std::cout << query_id << " ";
                        count++;
                    }
                    std::cout << "(showing first 3 of " << count << ")" << std::endl;
                }
            }
            
            std::string join_output_filename = result_dir + "/join_output.txt";
            // Remove existing output file to avoid appending to old results
            std::remove(join_output_filename.c_str());
            std::ofstream final_output(join_output_filename);
            
            if (!final_output.is_open()) {
                throw std::runtime_error("Could not open final output file: " + join_output_filename);
            }
            
            // First, read all cluster files once to get distance computation metrics
            std::vector<std::unordered_map<int, std::tuple<int, int, int>>> cluster_distance_metrics;
            cluster_distance_metrics.resize(g_cluster_windows.size());
            
            for (size_t cluster_id = 0; cluster_id < g_cluster_windows.size(); cluster_id++) {
                std::string cluster_output_filename = result_dir + "/cluster_" + std::to_string(cluster_id) + "_join_output.txt";
                if (std::filesystem::exists(cluster_output_filename)) {
                    std::ifstream cluster_file(cluster_output_filename);
                    std::string line;
                    while (std::getline(cluster_file, line)) {
                        std::istringstream iss(line);
                        int x_i, kappa_i, num_results;
                        double seed, neighbor, bfs, proc_time, memory;
                        iss >> x_i >> kappa_i >> num_results >> seed >> neighbor >> bfs >> proc_time >> memory;
                        
                        // Store (seed, neighbor, bfs) for this query in this cluster
                        cluster_distance_metrics[cluster_id][x_i] = std::make_tuple(
                            static_cast<int>(seed), 
                            static_cast<int>(neighbor), 
                            static_cast<int>(bfs)
                        );
                    }
                    cluster_file.close();
                }
            }
            
            // Post-process each query vector
            std::cout << "Post-processing " << supplier_vectors.size() << " query vectors..." << std::endl;
            for (const auto& [query_id, query_vec] : supplier_vectors) {
                if (query_vec.empty()) continue;
                
                // Collect all matches across all clusters for this query
                std::vector<int> all_matches;
                double total_processing_time = 0.0;
                int total_seed_distances = 0;
                int total_neighbor_distances = 0;
                int total_bfs_distances = 0;
                
                for (size_t cluster_id = 0; cluster_id < g_cluster_windows.size(); cluster_id++) {
                    const auto& windows = g_cluster_windows[cluster_id];
                    if (windows.count(query_id) == 0) {
                        continue;
                    }
                    const auto& window = windows.at(query_id);
                    
                    // For each local vector ID in the window, check distance and filter by epsilon
                    for (int local_vector_id : window) {
                        // Use local vectors directly (no conversion needed)
                        if (cluster_local_vectors_list[cluster_id].count(local_vector_id) > 0) {
                            // Convert local ID to global ID for triangle inequality pruning
                            int global_vector_id = -1;
                            if (local_to_global_mapping[cluster_id].count(local_vector_id) > 0) {
                                global_vector_id = local_to_global_mapping[cluster_id][local_vector_id];
                            }
                            
                            bool is_match = false;
                            double dist = 0.0;
                            
                            // Apply triangle inequality pruning if enabled
                            if (use_triangle_inequality && global_vector_id >= 0) {
                                // Get the correct cluster for this global_vector_id
                                auto [data_cluster_id, pos_in_cluster] = global_to_clusters2_mapping.at(global_vector_id);
                                auto [can_prune, is_match_pruned, dist_estimate] = apply_triangle_inequality_pruning(
                                    query_vec, global_vector_id, epsilon,
                                    clusters2_centroids, clusters2_centroid_distances,
                                    global_to_clusters2_mapping, cluster_vectors_list[0], //XXX all vectors are in the first cluster
                                    query_to_centroid_distances, query_id, "POSTPROCESSING");
                                
                                if (can_prune) {
                                    is_match = is_match_pruned;
                                    dist = dist_estimate;
                                } else {
                                    // Need to compute actual distance
                                    dist = l2_distance(query_vec, cluster_local_vectors_list[cluster_id][local_vector_id]);
                                    is_match = (dist <= epsilon);
                                }
                            } else {
                                // No triangle inequality pruning, compute distance normally
                                dist = l2_distance(query_vec, cluster_local_vectors_list[cluster_id][local_vector_id]);
                                is_match = (dist <= epsilon);
                            }
                            
                            // Add to results if it's a match
                            if (is_match && global_vector_id >= 0) {
                                all_matches.push_back(global_vector_id);
                            }
                        }
                    }
                    
                    // Add processing time for this cluster
                    total_processing_time += g_cluster_processing_times[cluster_id][query_id];
                    
                    // Add distance computations for this cluster from pre-loaded data
                    if (cluster_distance_metrics[cluster_id].count(query_id) > 0) {
                        auto [seed, neighbor, bfs] = cluster_distance_metrics[cluster_id][query_id];
                        total_seed_distances += seed;
                        total_neighbor_distances += neighbor;
                        total_bfs_distances += bfs;
                    }
                }
                
                // Write output line in same format as single-cluster: 
                // <x_i> <kappa_i> <num_join_results> <seed> <neighbor> <BFS> <processing_time_for_x_i> <memory_usage> <join_result_1> <join_result_2> ...
                // For multi-cluster: kappa_i = -1, aggregate all metrics across clusters
                double memory_usage = get_memory_usage_mb(); // Use current memory usage
                final_output << query_id << " " << -1 << " " << all_matches.size() << " " 
                           << total_seed_distances << " " << total_neighbor_distances << " " 
                           << total_bfs_distances << " " << total_processing_time << " " << memory_usage;
                for (int global_id : all_matches) {
                    final_output << " " << global_id;
                }
                final_output << std::endl;
                total_matches += all_matches.size();
            }
            
            final_output.close();
            std::cout << "Final results written to " << join_output_filename << " with " << total_matches << " total matches" << std::endl;
            
        } else {
            // Single cluster mode: original behavior
            std::cout << "Single cluster mode: processing original data" << std::endl;
            std::string join_output_filename = result_dir + "/join_output.txt";
            // Remove existing output file to avoid appending to old results
            std::remove(join_output_filename.c_str());
            
            // Initialize BFS data structures if collection is enabled
            if (collect_bfs_data) {
                g_greedy_visited.resize(supplier_vectors.size());
                g_bfs_seeds.resize(supplier_vectors.size());
                g_bfs_visited.resize(supplier_vectors.size());
                g_final_join_results.resize(supplier_vectors.size());
                std::cout << "BFS data collection enabled for " << supplier_vectors.size() << " queries" << std::endl;
            }
            
            // Load full mapping file here and parse into a mapping: new_id -> [original_ids]
            // Example format:
            //   --- MAPPING (new_id -> [original_ids]) ---
            //   3: 3 115186 275665 280053
            //   248937: 289955
            std::cout << "Loading full mapping from hardcoded path..." << std::endl;
            const std::string mapping_path = "nytimes/nytimes_base_mapping.txt"; // hardcoded path per requirement
            std::unordered_map<int, std::vector<int>> full_mapping;
            {
                std::ifstream mf(mapping_path);
                if (!mf.is_open()) {
                    std::cerr << "Warning: Could not open mapping file at " << mapping_path << ", proceeding without expansion" << std::endl;
                } else {
                    std::string line;
                    // Skip optional header line if present
                    std::streampos start_pos = mf.tellg();
                    if (std::getline(mf, line)) {
                        if (line.find("--- MAPPING") == std::string::npos) { mf.clear(); mf.seekg(start_pos); }
                    }
                    // Parse body lines
                    while (std::getline(mf, line)) {
                        if (line.empty()) continue;
                        // Allow lines with or without ':'
                        size_t colon_pos = line.find(':');
                        int key = -1;
                        std::vector<int> vals;
                        if (colon_pos != std::string::npos) {
                            // Left side is the key, right side has space-separated integers
                            try {
                                key = std::stoi(line.substr(0, colon_pos));
                            } catch (...) {
                                continue; // malformed key, skip
                            }
                            std::istringstream ss(line.substr(colon_pos + 1));
                            int v;
                            while (ss >> v) { vals.push_back(v); }
                        } else {
                            // Entire line space-separated: key v1 v2 ...
                            std::istringstream ls(line);
                            ls >> key;
                            if (ls.fail()) continue;
                            int v;
                            while (ls >> v) { vals.push_back(v); }
                        }
                        if (key >= 0 && !vals.empty()) {
                            full_mapping[key] = std::move(vals);
                        }
                    }
                    std::cout << "Parsed mapping entries: " << full_mapping.size() << std::endl;
                }
            }
            // get the supllier vecs out of the part vecs here (part vecs should still hold all vecs tho)
            sim_join(supplier_vectors, part_vectors, Gx, Gy, epsilon, w_queue, num_threads, mst_filename, false, join_output_filename, result_dir, force_kappa_y0, early_terminate, enable_top1_detection, enable_closest_fallback, -1,
                    use_triangle_inequality, 
                    use_triangle_inequality ? &clusters2_centroids : nullptr,
                    use_triangle_inequality ? &clusters2_centroid_distances : nullptr,
                    use_triangle_inequality ? &global_to_clusters2_mapping : nullptr,
                    use_triangle_inequality ? &cluster_vectors_list[0] : nullptr,
                    use_triangle_inequality ? &query_to_centroid_distances : nullptr,
                    nullptr,  // local_to_global_mapping not used in single-cluster mode
                    nullptr,  // query_to_centroid_distances_map not used in single-cluster mode
                    enable_query_to_data_edges,
                    k_top_data_points,
                    data_offset,
                    collect_bfs_data,
                    break_before_bfs,
                    seed_offset,
                    enable_seed_offset_filtering,
                    sort_jj_by_distance,
                    cache_closest_only, 
                    part_entry_point, 
                    max_bfs_out_of_range_tolerance,
                    adaptive_bfs_threshold_factor,
                    one_hop_data_only,
                    topK,
                    patience,
                    ood_flags,
                    number_cached,
                    merged_index
                );
                    // (full_mapping.empty() ? nullptr : &full_mapping));
            
            // Write BFS data to files if collection is enabled
            if (collect_bfs_data) {
                std::string seeds_filename = result_dir + "/seeds.txt";
                std::string bfs_filename = result_dir + "/BFS.txt";
                std::string greedy_filename = result_dir + "/greedy.txt";
                std::string join_results_filename = result_dir + "/J.txt";
                
                std::ofstream seeds_file(seeds_filename);
                std::ofstream bfs_file(bfs_filename);
                std::ofstream greedy_file(greedy_filename);
                std::ofstream join_results_file(join_results_filename);
                
                if (seeds_file.is_open() && bfs_file.is_open() && greedy_file.is_open() && join_results_file.is_open()) {
                    for (size_t query_id = 0; query_id < g_bfs_seeds.size(); query_id++) {
                        // Write seeds for this query
                        for (size_t i = 0; i < g_bfs_seeds[query_id].size(); i++) {
                            if (i > 0) seeds_file << " ";
                            seeds_file << g_bfs_seeds[query_id][i];
                        }
                        seeds_file << std::endl;
                        
                        // Write visited nodes with depth for this query
                        for (size_t i = 0; i < g_bfs_visited[query_id].size(); i++) {
                            if (i > 0) bfs_file << " ";
                            bfs_file << g_bfs_visited[query_id][i].first << " " << g_bfs_visited[query_id][i].second;
                        }
                        bfs_file << std::endl;
                        
                        // Write greedy visited nodes with depth for this query
                        for (size_t i = 0; i < g_greedy_visited[query_id].size(); i++) {
                            if (i > 0) greedy_file << " ";
                            greedy_file << g_greedy_visited[query_id][i].first << " " << g_greedy_visited[query_id][i].second;
                        }
                        greedy_file << std::endl;
                        
                        // Write final join results for this query
                        for (size_t i = 0; i < g_final_join_results[query_id].size(); i++) {
                            if (i > 0) join_results_file << " ";
                            join_results_file << g_final_join_results[query_id][i];
                        }
                        join_results_file << std::endl;
                    }
                    
                    seeds_file.close();
                    bfs_file.close();
                    greedy_file.close();
                    join_results_file.close();
                    std::cout << "BFS data written to " << seeds_filename << " and " << bfs_filename << std::endl;
                    std::cout << "Greedy visited nodes written to " << greedy_filename << std::endl;
                    std::cout << "Final join results written to " << join_results_filename << std::endl;
                } else {
                    std::cerr << "Error: Could not create BFS output files" << std::endl;
                }
            }
            
            // Count total results for single cluster mode from output file
            std::ifstream result_file(join_output_filename);
            if (result_file.is_open()) {
                std::string line;
                while (std::getline(result_file, line)) {
                    std::istringstream iss(line);
                    int x_i, kappa_i, num_results;
                    double seed, neighbor, bfs, proc_time, memory;
                    iss >> x_i >> kappa_i >> num_results >> seed >> neighbor >> bfs >> proc_time >> memory;
                    total_matches += num_results;
                }
                result_file.close();
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto join_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
        
        std::cout << "sim_join completed successfully" << std::endl;
        
        // Check if shutdown was requested during execution
        if (g_shutdown_requested) {
            std::cout << "Program was interrupted during execution." << std::endl;
        }
        
        std::cout << "\nResults:" << std::endl;
        std::cout << "Join time: " << join_time << " seconds" << std::endl;
        std::cout << "Total join results: " << total_matches << std::endl;

        assert(cluster_vectors_list.size() == 1);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::cerr << "This might be due to incorrect file paths or data format." << std::endl;
        std::cerr << "Please check that the embedding files exist and have the correct format." << std::endl;
        return 1;
    }
    
    // Print profiling statistics at the end
    g_profile_data.print_stats();
    
    // Save profiling results to file
    std::string profiling_filename = result_dir + "/profiling_results.txt";
    g_profile_data.save_stats_to_file(profiling_filename);
    
    return 0;
} 
