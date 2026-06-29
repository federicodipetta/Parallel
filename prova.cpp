#include <iostream>
#include <limits>
#include <math.h>
#include <iomanip>
#include <cstring>
#define T 1e-5
using namespace std;

inline float calc_distance(float *point, float *centroids, int fratures)
{
    float distance = 0;
    for (int i = 0; i < fratures; ++i)
    {
        float diff = point[i] - centroids[i];
        distance += diff * diff;
    }
    // avoiding SQRT for performance
    return distance;
}
/*
    1,2 
    2,3

    1  3 = 3
    2  5 = 5
    2
    3
*/
void k_means(float *points, float *centroids, int *labels, size_t num_points, size_t num_features, size_t num_cetroids, size_t max_epochs)
{
    int epochs = 0;
    float error = std::numeric_limits<float>::max();
    float prev_error = error;
    float *new_centroids = (float *)malloc(sizeof(*centroids) * num_cetroids * num_features);
    int *counts = (int *)malloc(sizeof(*counts) * num_cetroids);
    do
    {
        memset(counts, 0, sizeof(*counts) * num_cetroids);
        memset(new_centroids, 0, sizeof(*new_centroids) * num_cetroids * num_features);

        prev_error = error;
        error = 0;
        // 1. Calculate for each point the error
        size_t i, j, h;
        for (size_t i = 0; i < num_points; ++i)
        {
            // 1.1 Find the nearest cenrtroid
            float min_distance = std::numeric_limits<float>::max();
            for (size_t j = 0; j < num_cetroids; ++j)
            {
                float distance = calc_distance(points + (i * num_features), centroids + (j * num_features), num_features);
                if (distance < min_distance)
                {
                    // set new cluster for the point h
                    labels[i] = j;
                    min_distance = distance;
                }
            }
            // Updates cluster
            counts[labels[i]]++;

            for (h = 0; h < num_features; ++h)
                new_centroids[labels[i] * num_features + h] += points[i * num_features + h];

            error += min_distance;
        } // END POINTS LOOP
        epochs++;
        for (j = 0; j < num_cetroids; ++j)
            if (counts[j] > 0)
                for (h = 0; h < num_features; ++h)
                    centroids[j * num_features + h] = new_centroids[j * num_features + h] / counts[j];

    } while (max_epochs > epochs && (fabs(prev_error - error)) > T);
    // unecessary for short executions
    free(new_centroids);
    free(counts);
}

int main(void)
{
    int num_points = 1000;
    int features = 2;
    int num_clusters = 3;
    int max_epochs = 100;
    float *points = new float[num_points * features];
    float *centroids = new float[num_clusters * features];
    int *labels = new int[num_points];

    // initialize centroids and points with random values

    for (int i = 0; i < num_points * features; i++)
    {
        points[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < num_clusters * features; i++)
    {
        centroids[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < num_points; i++)
    {
        labels[i] = -1;
    }
    int *cluster_sizes = new int[num_clusters]();
    cout << "\n=== INITIAL ===\n";

    for (int c = 0; c < num_clusters; ++c)
    {
        cout << "Cluster " << c << ": (";

        for (int f = 0; f < features; ++f)
        {
            cout << fixed << setprecision(4)
                 << centroids[c * features + f];

            if (f < features - 1)
                cout << ", ";
        }

        cout << ")\n";
    }

    cout << "\n=== CLUSTER SIZES ===\n";

    cluster_sizes = new int[num_clusters]();

    for (int i = 0; i < num_points; ++i)
        cluster_sizes[labels[i]]++;

    for (int c = 0; c < num_clusters; ++c)
    {
        cout << "Cluster "
             << c
             << " -> "
             << cluster_sizes[c]
             << " points\n";
    }

    cout << "\n=== SAMPLE ASSIGNMENTS ===\n";

    for (int i = 0; i < min(20, num_points); ++i)
    {
        cout
            << setw(4) << i
            << " : ("
            << fixed << setprecision(4)
            << points[i * features]
            << ", "
            << points[i * features + 1]
            << ") -> Cluster "
            << labels[i]
            << '\n';
    }

    k_means(points, centroids, labels, num_points, features, num_clusters, max_epochs);

    cout << "\n=== FINAL CENTROIDS ===\n";

    for (int c = 0; c < num_clusters; ++c)
    {
        cout << "Cluster " << c << ": (";

        for (int f = 0; f < features; ++f)
        {
            cout << fixed << setprecision(4)
                 << centroids[c * features + f];

            if (f < features - 1)
                cout << ", ";
        }

        cout << ")\n";
    }

    cout << "\n=== CLUSTER SIZES ===\n";

    cluster_sizes = new int[num_clusters]();

    for (int i = 0; i < num_points; ++i)
        cluster_sizes[labels[i]]++;

    for (int c = 0; c < num_clusters; ++c)
    {
        cout << "Cluster "
             << c
             << " -> "
             << cluster_sizes[c]
             << " points\n";
    }

    cout << "\n=== SAMPLE ASSIGNMENTS ===\n";

    for (int i = 0; i < min(20, num_points); ++i)
    {
        cout
            << setw(4) << i
            << " : ("
            << fixed << setprecision(4)
            << points[i * features]
            << ", "
            << points[i * features + 1]
            << ") -> Cluster "
            << labels[i]
            << '\n';
    }

    delete[] cluster_sizes;
    return 0;
}