#ifndef __KMEANS_HPP__
#define  __KMEANS_HPP__

#include <tuple>
#include <vector>
#include <cmath>
#include <limits>
#include <array>
#include <iostream>
#include <fstream>
#include <cstring>
#include <chrono>
#include <omp.h>



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
        kmeans(std::vector<Point<dim>>& dataset, unsigned int K){
    
    size_t n_points {dataset.size()};
   
    if(n_points < K){
        std::cerr << "Not enough points in the dataset for the required number of centres.\n";
        throw std::exception();
    }else if(K == 0){
        std::cerr << "Invalid K value (0)\n";
        throw std::exception();
    }
  
    std::vector<Point<dim>> centres;
    for(int i {0}; i < K; i++) centres.push_back(dataset[i]);

    std::vector<int> assignment(dataset.size(), -1);
    double init_value[dim] {};
    bool converged = true;
  
    // temporary storage
    std::vector<size_t> counters (K, 0);
    std::cout << "Computation starts..." << std::endl;
    #pragma omp parallel shared(converged)
    {
    do{
        #pragma omp barrier
       
        #pragma omp single
        converged = true;
        
        bool locally_converged = true;

        #pragma omp for schedule(static) nowait
        for(size_t i = 0; i < dataset.size(); i++){
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
            if(chosen_centre != assignment[i]){
                assignment[i] = chosen_centre;
                locally_converged = false;
            }
        }
      
        #pragma omp critical
        converged = (converged && locally_converged);
    
        #pragma omp barrier

        if(!converged){        
            // recompute centres locally
            std::vector<Point<dim>> local_centres (K);
            std::vector<size_t> local_counters (K, 0);
            #pragma omp for schedule(static) nowait
            for(size_t i = 0u; i < assignment.size(); i++){
                int c {assignment[i]};
                local_centres[c] += dataset[i];
                local_counters[c] += 1u;
            }

            // compute final centres
            #pragma omp single
            for(size_t i = 0u; i < centres.size(); i++){
                centres[i] = Point<dim>(init_value);
                counters[i] = 0;
            }
            #pragma omp critical
            for(size_t i = 0u; i < centres.size(); i++){
                centres[i] += local_centres[i];
                counters[i] += local_counters[i];
            }
            #pragma omp barrier
            #pragma omp single
            for(size_t i {0u}; i < centres.size(); i++){
                centres[i] /= counters[i];
            }
        }
    }while(!converged);
    }
    return {centres, assignment};
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

