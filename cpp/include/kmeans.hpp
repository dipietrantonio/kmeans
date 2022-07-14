#ifndef __KMEANS_HPP__
#define  __KMEANS_HPP__

#include <tuple>
#include <vector>
#include <cmath>
#include <limits>
#include <array>
#include <iostream>
#include <fstream>
#include <mpi.h>
#include <cstring>
#include <chrono>


char MPI_error_buffer[512];
int length_of_error_string;

#define MPI_CHECK_ERROR(X)({\
    if((X) != MPI_SUCCESS){\
        MPI_Error_string((X), MPI_error_buffer, &length_of_error_string);\
        fprintf(stderr, "MPI error: %s\n", MPI_error_buffer);\
    }\
})


template <unsigned int dim>
class Point {

    protected:
    double coord[dim];

    public:
    
    Point(){
        memset(coord, 0, sizeof(double) * dim);
    };


    Point(double *coord){
        for(unsigned int i {0u}; i < dim; i++)
            this->coord[i] = coord[i];
    };


    bool operator==(const Point<dim>& other) const {
        for (unsigned int i {0u}; i < dim; i++){
            if(std::abs(this->coord[i] - other.coord[i]) > std::numeric_limits<float>::epsilon()){
                return false;
            }
        }
        return true;
    };


    friend std::ostream& operator<<(std::ostream& os, const Point<dim>& p){
        os << "(";
        for(unsigned int d {0}; d < dim - 1; d++)
            os << p.coord[d] << ", ";
        os << p.coord[dim-1] << ")";
        return os; 
    }


    void operator+=(const Point<dim>& other){
        for(unsigned int i {0u}; i < dim; i++){
            coord[i] += other.coord[i];
        }
    }


    template <typename T>
    void operator/=(const T& other){
        for(unsigned int i {0u}; i < dim; i++){
            coord[i] /= other;
        }
    }


    float distance(const Point<dim> &other) const {
        float dist  {0.0f};
        for (unsigned int i {0u}; i < dim; i++)
            dist += powf(this->coord[i] - other.coord[i], 2);
        return sqrt(dist);
    };
};


template <unsigned int dim>
void scatter_dataset(const std::vector<Point<dim>>& dataset_in, std::vector<Point<dim>>& dataset_out, size_t& dataset_size){
    int world_size, world_rank;
    MPI_CHECK_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    
    unsigned int n_points = 0;    
    if(world_rank == 0) n_points = dataset_in.size();
    MPI_CHECK_ERROR(MPI_Bcast(&n_points, 1, MPI_INT, 0, MPI_COMM_WORLD));
    dataset_size = n_points;

    int max_points_per_process = ((n_points + world_size - 1) / world_size) * sizeof(Point<dim>);
    int send_counts[world_size];
    int displs[world_size];
    size_t remaining_items {n_points * sizeof(Point<dim>)};
    displs[0] = 0;
    send_counts[0] = max_points_per_process < remaining_items ? max_points_per_process : remaining_items;
    for (int i {1}; i < world_size; i++){
        remaining_items -= send_counts[i-1];
        send_counts[i] = max_points_per_process < remaining_items ? max_points_per_process : remaining_items;
        displs[i] = displs[i-1] + send_counts[i-1];
    }
    size_t n_points_local {send_counts[world_rank]/sizeof(Point<dim>)};
    dataset_out.resize(n_points_local);
    MPI_CHECK_ERROR(MPI_Scatterv(dataset_in.data(), send_counts, displs, MPI_CHAR, (char*) dataset_out.data(), send_counts[world_rank], MPI_CHAR, 0, MPI_COMM_WORLD));   
}


template <unsigned int dim>
void gather_centres(const std::vector<Point<dim>>& dataset, unsigned int K, std::vector<Point<dim>>& centres){
    int world_size, world_rank;
    MPI_CHECK_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    centres.resize(K);
    int max_centres_per_process = ((static_cast<int>(K) + world_size - 1) / world_size) * sizeof(Point<dim>);
    int send_counts[world_size];
    int displs[world_size];
    size_t remaining_items {K * sizeof(Point<dim>)};
    displs[0] = 0;
    send_counts[0] = max_centres_per_process < remaining_items ? max_centres_per_process : remaining_items;
    for (int i {1}; i < world_size; i++){
        remaining_items -= send_counts[i-1];
        send_counts[i] = max_centres_per_process < remaining_items ? max_centres_per_process : remaining_items;
        displs[i] = displs[i-1] + send_counts[i-1];
    }
    MPI_CHECK_ERROR(MPI_Allgatherv(dataset.data(), send_counts[world_rank], MPI_CHAR, centres.data(), send_counts, displs,
        MPI_CHAR, MPI_COMM_WORLD));
}


/**
 * Performs k-means clustering on a dataset using K centres.
 * @param dataset a vector of points.
 * @param K number of centres.
 * @returns A tuple where the first element is the vector of centres
 * and the second one is the assignment of points to centres.
 */
template <unsigned int dim>
std::tuple<std::vector<Point<dim>>, std::vector<int>>
        kmeans(std::vector<Point<dim>>& dataset, unsigned int K){
    
    std::vector<Point<dim>> local_dataset;
    size_t n_points;
    scatter_dataset(dataset, local_dataset, n_points);

    if(n_points < K){
        std::cerr << "Not enough points in the dataset for the required number of centres.\n";
        throw std::exception();
    }else if(K == 0){
        std::cerr << "Invalid K value (0)\n";
        throw std::exception();
    }
    auto start = std::chrono::steady_clock::now();
    int world_size, world_rank;
    MPI_CHECK_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    // set initial centers
    std::vector<Point<dim>> centres;
    gather_centres(local_dataset, K, centres);

    std::vector<int> assignment(local_dataset.size(), -1);
    double init_value[dim] {};
    bool converged;
    bool *all_converged = new bool[world_size];
    // temporary storage
    std::vector<size_t> counters (K, 0);
    do{
        converged = true;
        #pragma omp parallel for schedule(static)
        for(size_t i = 0; i < local_dataset.size(); i++){
            Point<dim>& p {local_dataset[i]};
            float best_dist {p.distance(centres[0])};
            int chosen_centre {0};
            for(int j {1}; j < centres.size(); j++){
                float dist {p.distance(centres[j])};
                if(dist < best_dist){
                    best_dist = dist;
                    chosen_centre = j;
                }
            }
            if(chosen_centre != assignment[i]){
                assignment[i] = chosen_centre;
                converged = false;
            }
        }

        MPI_CHECK_ERROR(MPI_Allgather(&converged, 1, MPI_C_BOOL, all_converged, 1,
            MPI_C_BOOL, MPI_COMM_WORLD));
        converged = true;
        for(unsigned int p {0}; p < world_size; p++)
            if(!all_converged[p]) {
                converged = false;
                break;
            }
        if(converged){
            // gather assignment
            std::vector<int> all_assignment(n_points);
            int *counts = new int[world_size];
            int *displ = new int[world_size];
            int remaining_items {n_points};
            size_t items_unit {(n_points + world_size - 1) / world_size};
            counts[0] = items_unit < remaining_items ? items_unit : (remaining_items);
            displ[0] = 0;
            for(int r {1}; r < world_size; r++){
                remaining_items -= counts[r-1];
                counts[r] = items_unit < remaining_items ? items_unit : (remaining_items);
                displ[r] = displ[r-1] + counts[r-1];
            }
            MPI_CHECK_ERROR(MPI_Allgatherv(assignment.data(), assignment.size(), MPI_INT, all_assignment.data(),
                counts, displ, MPI_INT, MPI_COMM_WORLD));
            delete[] counts;
            delete[] displ;
            delete[] all_converged;
            auto stop = std::chrono::steady_clock::now();
            std::cout << "Kmeans execution time: " << std::chrono::duration_cast<std::chrono::seconds>(stop-start).count() << std::endl;
            return {centres, all_assignment};
        }
        // recompute centres
        for(auto& c : centres)
            c = Point<dim>(init_value);
        for(size_t &c : counters)
            c = 0;
        for(size_t i {0}; i < assignment.size(); i++){
            int c {assignment[i]};
            centres[c] += local_dataset[i];
            counters[c] += 1u;
        }
        // exchange centers and counters
        std::vector<size_t> all_counters (K * world_size);
        std::vector<Point<dim>> all_centers (K * world_size);
        MPI_CHECK_ERROR(MPI_Allgather(centres.data(), sizeof(Point<dim>) * K, MPI_CHAR, all_centers.data(),
            sizeof(Point<dim>) * K, MPI_CHAR, MPI_COMM_WORLD));
        MPI_CHECK_ERROR(MPI_Allgather(counters.data(), sizeof(size_t) * K, MPI_CHAR, all_counters.data(),
            sizeof(size_t) * K, MPI_CHAR, MPI_COMM_WORLD));
        for(unsigned int rank {0u}; rank < world_size; rank++){
            if(rank == world_rank) continue;
            for(unsigned int k {0u}; k < K; k++){
                counters[k] += all_counters[K*rank + k];
                centres[k] += all_centers[K*rank + k];
            }
        }
        for(size_t i {0u}; i < centres.size(); i++){
            centres[i] /= counters[i];
        }
    }while(true); 
}

template <unsigned int dim>
void read_dataset(std::vector<Point<dim>>& dataset, const char *filename){
    unsigned int n_points;
    std::ifstream in {filename};
    in >> n_points;
    dataset.resize(n_points);
    double coord[dim];
    for(unsigned int i {0u}; i < n_points; i++){
        for(unsigned int j {0u}; j < dim; j++){
            in >> coord[j];
        }
        dataset[i] = Point<dim>(coord);
    }
    in.close();
}

#endif

