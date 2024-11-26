// HamiltonianPath.cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <tuple>
#include <set>
#include <limits>
#include <thread>
#include <mutex>
#include <future>
#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

// Type Definitions
using Point = std::pair<double, double>;
using Path = std::vector<int>;
using DistanceMatrix = std::vector<std::vector<double>>;
using Steps = std::vector<Path>;

// Global Logger Variables
std::ofstream log_file;
std::mutex log_mutex;

// Logger Function
void log(const std::string& message) {
    std::lock_guard<std::mutex> lock(log_mutex);
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    // Format time
    std::tm buf;
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif
    log_file << std::put_time(&buf, "%Y-%m-%d %X") << " - " << message << std::endl;
}

// Function to parse a .tsp file (TSPLIB format)
bool parse_tsp_file(const std::string& filename, std::vector<Point>& points) {
    log("Starting to parse .tsp file: " + filename);
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        log("Error: Unable to open the file " + filename);
        std::cerr << "Error: Unable to open the file " << filename << std::endl;
        return false;
    }

    std::string line;
    bool reading_nodes = false;
    while (std::getline(infile, line)) {
        if (line.find("NODE_COORD_SECTION") != std::string::npos) {
            reading_nodes = true;
            log("Found NODE_COORD_SECTION");
            continue;
        }
        if (line.find("EOF") != std::string::npos) {
            log("Reached EOF");
            break;
        }
        if (reading_nodes) {
            std::istringstream iss(line);
            int index;
            double x, y;
            if (!(iss >> index >> x >> y)) {
                log("Warning: Skipping invalid line: " + line);
                continue; // Skip invalid lines
            }
            // Adjust index to 0-based if necessary
            // Assuming .tsp file uses 0-based indexing
            // If .tsp uses 1-based, uncomment the next line
            // index -= 1;
            points.emplace_back(x, y);
            log("Parsed point " + std::to_string(index) + ": (" + std::to_string(x) + ", " + std::to_string(y) + ")");
        }
    }

    infile.close();
    if (points.empty()) {
        log("Error: No points found in the file " + filename);
        std::cerr << "Error: No points found in the file " << filename << std::endl;
        return false;
    }
    log("Completed parsing .tsp file. Total points: " + std::to_string(points.size()));
    return true;
}

// Function to compute Euclidean distance matrix
DistanceMatrix compute_distance_matrix(const std::vector<Point>& points) {
    log("Starting computation of distance matrix");
    int n = points.size();
    DistanceMatrix distance_matrix(n, std::vector<double>(n, 0.0));
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            if(i != j){
                double dx = points[i].first - points[j].first;
                double dy = points[i].second - points[j].second;
                distance_matrix[i][j] = std::sqrt(dx*dx + dy*dy);
            }
        }
    }
    log("Completed computation of distance matrix");
    return distance_matrix;
}

// Function to calculate total path distance
double total_path_distance(const Path& path, const DistanceMatrix& distance_matrix) {
    double total = 0.0;
    for(size_t i = 0; i < path.size() -1; ++i){
        total += distance_matrix[path[i]][path[i+1]];
    }
    return total;
}

// Function to build path incrementally
std::pair<Path, Steps> build_path_incremental(Path current_path, std::set<int> remaining_points, 
                                             const DistanceMatrix& distance_matrix) {
    Steps steps;
    steps.emplace_back(current_path);
    while(!remaining_points.empty()){
        int best_r = -1;
        int best_insertion_position = -1;
        double best_delta_distance = std::numeric_limits<double>::infinity();

        for(auto r : remaining_points){
            for(int i = 0; i <= static_cast<int>(current_path.size()); ++i){
                double delta = 0.0;
                if(i == 0){
                    // Inserting at the beginning
                    delta = distance_matrix[r][current_path[0]];
                }
                else if(i == static_cast<int>(current_path.size())){
                    // Inserting at the end
                    delta = distance_matrix[current_path.back()][r];
                }
                else{
                    // Inserting in the middle
                    int p = current_path[i-1];
                    int q = current_path[i];
                    delta = distance_matrix[p][r] + distance_matrix[r][q] - distance_matrix[p][q];
                }

                if(delta < best_delta_distance){
                    best_delta_distance = delta;
                    best_r = r;
                    best_insertion_position = i;
                }
            }
        }

        if(best_r != -1){
            current_path.insert(current_path.begin() + best_insertion_position, best_r);
            remaining_points.erase(best_r);
            steps.emplace_back(current_path);
        }
        else{
            break;
        }
    }
    return {current_path, steps};
}

// Function to build path with lookahead
std::pair<Path, Steps> build_path_least_distance_updated(Path start_edge, std::set<int> remaining_points, 
                                                         const DistanceMatrix& distance_matrix) {
    Steps steps;
    Path path = start_edge;
    steps.emplace_back(path);
    while(!remaining_points.empty()){
        int best_r = -1;
        int best_insertion_position = -1;
        double best_total_distance = std::numeric_limits<double>::infinity();

        for(auto r : remaining_points){
            for(int i = 0; i <= static_cast<int>(path.size()); ++i){
                // Create a temporary path with r inserted at position i
                Path temp_path = path;
                temp_path.insert(temp_path.begin() + i, r);

                // Create a temporary remaining set
                std::set<int> temp_remaining = remaining_points;
                temp_remaining.erase(r);

                // Simulate building the path incrementally
                auto [simulated_path, simulated_steps] = build_path_incremental(temp_path, temp_remaining, distance_matrix);
                double simulated_distance = total_path_distance(simulated_path, distance_matrix);

                if(simulated_distance < best_total_distance){
                    best_total_distance = simulated_distance;
                    best_r = r;
                    best_insertion_position = i;
                }
            }
        }

        if(best_r != -1){
            path.insert(path.begin() + best_insertion_position, best_r);
            remaining_points.erase(best_r);
            steps.emplace_back(path);
        }
        else{
            break;
        }
    }
    return {path, steps};
}

// Function to process a single edge for parallel execution
std::pair<double, Path> process_edge(int edge_start, int edge_end, const DistanceMatrix& distance_matrix, int n, int edge_id) {
    // Log the start of edge processing
    log("Processing edge " + std::to_string(edge_id) + ": (" + std::to_string(edge_start) + ", " + std::to_string(edge_end) + ")");

    // Initialize path with the edge
    Path start_edge = {edge_start, edge_end};
    std::set<int> remaining_points;
    for(int i = 0; i < n; ++i){
        if(i != edge_start && i != edge_end){
            remaining_points.insert(i);
        }
    }

    // Build path using the heuristic
    auto [path, steps] = build_path_least_distance_updated(start_edge, remaining_points, distance_matrix);

    if(path.size() == n){
        double distance = total_path_distance(path, distance_matrix);
        log("Edge " + std::to_string(edge_id) + " completed. Total Distance: " + std::to_string(distance));
        return {distance, path};
    }
    // Log failure to find a complete path
    log("Edge " + std::to_string(edge_id) + " failed to find a complete Hamiltonian path.");
    return {std::numeric_limits<double>::infinity(), Path()};
}

int main(int argc, char* argv[]) {
    // Open log file
    log_file.open("program.log", std::ios::out);
    if (!log_file.is_open()) {
        std::cerr << "Error: Unable to open log file for writing." << std::endl;
        return 1;
    }
    log("Program started.");

    // Check for .tsp file argument
    if(argc < 2){
        log("Error: No .tsp file provided as an argument.");
        std::cerr << "Usage: " << argv[0] << " <filename.tsp>" << std::endl;
        log("Program terminated due to missing arguments.");
        return 1;
    }

    std::string filename = argv[1];
    std::vector<Point> points;

    // Parse the .tsp file
    if(!parse_tsp_file(filename, points)){
        log("Program terminated due to file parsing error.");
        log_file.close();
        return 1;
    }

    int n = points.size();
    log("Number of points: " + std::to_string(n));
    std::cout << "Number of points: " << n << std::endl;

    // Compute distance matrix
    log("Starting computation of distance matrix.");
    std::cout << "Computing distance matrix..." << std::endl;
    DistanceMatrix distance_matrix = compute_distance_matrix(points);
    log("Completed computation of distance matrix.");
    std::cout << "Distance matrix computed." << std::endl;

    // Generate all possible edges (combinations of two points)
    log("Generating all possible edges.");
    std::vector<std::pair<int, int>> edges;
    edges.reserve(n * (n -1) / 2);
    for(int i = 0; i < n; ++i){
        for(int j = i +1; j < n; ++j){
            edges.emplace_back(i, j);
        }
    }
    log("Total number of edges to process: " + std::to_string(edges.size()));
    std::cout << "Total number of edges to process: " << edges.size() << std::endl;

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();
    log("Starting parallel processing of edges.");

    // Vector to hold futures
    std::vector<std::future<std::pair<double, Path>>> futures;
    futures.reserve(edges.size());

    // Launch asynchronous tasks for each edge
    for(size_t idx = 0; idx < edges.size(); ++idx){
        int edge_id = static_cast<int>(idx) +1;
        futures.emplace_back(std::async(std::launch::async, process_edge, edges[idx].first, edges[idx].second, std::cref(distance_matrix), n, edge_id));
    }

    // Collect results
    log("Collecting results from all edges.");
    std::vector<std::pair<double, Path>> complete_paths;
    complete_paths.reserve(edges.size());

    for(size_t idx = 0; idx < futures.size(); ++idx){
        auto result = futures[idx].get();
        if(result.first < std::numeric_limits<double>::infinity()){
            complete_paths.emplace_back(result);
            log("Edge " + std::to_string(idx +1) + ": Complete path found with distance " + std::to_string(result.first));
        }
    }

    // End timer
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    log("Completed parallel processing of edges.");
    log("Total computation time: " + std::to_string(elapsed.count()) + " seconds");
    std::cout << "\nTotal computation time: " << elapsed.count() << " seconds" << std::endl;

    if(!complete_paths.empty()){
        // Find the path with the minimum distance
        auto best_result = std::min_element(complete_paths.begin(), complete_paths.end(),
            [](const std::pair<double, Path>& a, const std::pair<double, Path>& b) -> bool {
                return a.first < b.first;
            });

        double best_distance = best_result->first;
        Path best_path = best_result->second;

        // Convert to 0-based indexing for output
        std::cout << "\nBest Hamiltonian Path:" << std::endl;
        log("Best Hamiltonian Path found with distance " + std::to_string(best_distance));
        for(auto city : best_path){
            std::cout << city << " "; // No +1 for 0-based indexing
        }
        std::cout << std::endl;
        std::cout << "Total Distance: " << best_distance << std::endl;
        log("Best Hamiltonian Path: ");
        for(auto city : best_path){
            log(std::to_string(city));
        }
        log("Total Distance: " + std::to_string(best_distance));
    }
    else{
        std::cout << "\nNo complete Hamiltonian path was found." << std::endl;
        log("No complete Hamiltonian path was found.");
    }

    log("Program terminated successfully.");
    log_file.close();
    return 0;
} 