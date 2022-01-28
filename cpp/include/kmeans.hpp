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

/**
 * Performs k-means clustering on a dataset using K centres.
 * @param dataset a vector of points.
 * @param K number of centres.
 * @returns A tuple where the first element is the vector of centres
 * and the second one is the assignment of points to centres.
 */
template <unsigned int dim>
std::tuple<std::vector<Point<dim>>, std::vector<int>>
        kmeans (std::vector<Point<dim>>& dataset, unsigned int K){
    if(dataset.size() < K){
        std::cerr << "Not enough points in the dataset for the required number of centres.\n";
        throw std::exception();
    }else if(K == 0){
        std::cerr << "Invalid K value (0)\n";
        throw std::exception();
    }
    
    int world_size, world_rank;
    MPI_CHECK_ERROR(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    MPI_CHECK_ERROR(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    // set initial centers
    std::vector<Point<dim>> centres;
    if(world_rank == 0){
        for(unsigned int i {0}; i < K; i++)
            centres.push_back(dataset[i]);
    }else{
        centres.resize(K);
    }
    MPI_CHECK_ERROR(MPI_Bcast(centres.data(), sizeof(Point<dim>)*K, MPI_CHAR, 0, MPI_COMM_WORLD));
    // partition dataset
    size_t n_local_points, start, stop;
    if(dataset.size() >= world_size){
        n_local_points = (dataset.size() + world_size - 1) / world_size;
        start = n_local_points * world_rank;
        stop = (world_rank == world_size - 1) ? dataset.size() : (n_local_points * (world_rank + 1));
    }else if(world_rank < dataset.size()){
        n_local_points = 1;
        start = world_rank;
        stop = start + 1;
    }else{
        n_local_points = start = stop = 0;
    }
    
    std::vector<int> assignment(stop - start, -1);
    double init_value[dim] {};
    bool converged;
    bool *all_converged = new bool[world_size];
    // temporary storage
    std::vector<size_t> counters (K, 0);
    do{

        if(world_rank == 0) std::cout << "centres: " << centres[0] << ", " << centres[1] << ";" << std::endl;
        converged = true;
        for(size_t i {start}; i < stop; i++){
            Point<dim>& p {dataset[i]};
            float best_dist {p.distance(centres[0])};
            int chosen_centre {0};
            for(int j {1}; j < centres.size(); j++){
                float dist {p.distance(centres[j])};
                if(dist < best_dist){
                    best_dist = dist;
                    chosen_centre = j;
                }
            }
            if(chosen_centre != assignment[i-start]){
                assignment[i-start] = chosen_centre;
                converged = false;
            }
        }

        //std::cout << "process " << world_rank << " converged: " << converged << std::endl;
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
            std::vector<int> all_assignment(dataset.size());
            int *counts = new int[world_size];
            int *displ = new int[world_size];
            int remaining_items {dataset.size()};
            size_t items_unit {(dataset.size() + world_size - 1) / world_size};
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
            return {centres, all_assignment};
        }
        // recompute centres
        for(auto& c : centres)
            c = Point<dim>(init_value);
        for(size_t &c : counters)
            c = 0;
        for(size_t i {start}; i < stop; i++){
            int c {assignment[i-start]};
            centres[c] += dataset[i];
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
    std::ifstream in {filename};
    unsigned int n_points;
    in >> n_points;
    double coord[dim];
    for(unsigned int i {0u}; i < n_points; i++){
        in >> coord[0];
        in >> coord[1];
        dataset.push_back(Point<dim>(coord));
    }
    in.close();
}

#endif

