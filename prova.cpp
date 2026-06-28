#include <iostream>
#include <limits>
#include <math.h>
#include <iomanip>
#include <cstring>
#define T 1e-5
using namespace std;

float calc_distance(float *point, float *centroids, int fratures)
{
    float distance = 0;
    int i;
    for (int i = 0; i < fratures; ++i)
    {
        float diff = point[i] - centroids[i];
        distance += diff * diff;
    }
    // avoiding SQRT for performance
    return distance;
}

void k_means(float *points, float *centroids, int *labels, int num_points, size_t features, size_t num_clusters, size_t max_epochs)
{
    int epochs = 0;
    float error = std::numeric_limits<float>::max();
    float prev_error = error;
    float *new_centroids = (float *)malloc(sizeof(*centroids) * num_clusters * features);
    int *counts = (int *)malloc(sizeof(*counts) * num_clusters);
    do
    {
        memset(counts, 0, sizeof(*counts) * num_clusters);
        memset(new_centroids, 0, sizeof(*new_centroids) * num_clusters * features);

        prev_error = error;
        error = 0;
        // 1. Calculate for each point the error
        int i, j, h;
        for (int i = 0; i < num_points; ++i)
        {
            // 1.1 Find the nearest cenrtroid
            float min_distance = std::numeric_limits<float>::max();
            for (int j = 0; j < num_clusters; ++j)
            {
                float distance = calc_distance(points + (i * features), centroids + (j * features), features);
                if (distance < min_distance)
                {
                    // set new cluster for the point h
                    labels[i] = j;
                    min_distance = distance;
                }
            }
            // Updates cluster
            counts[labels[i]]++;
            for (h = 0; h < features; ++h)
                new_centroids[labels[i] * features + h] += points[i * features + h];

            error += min_distance;
        }
        epochs++;
        for (j = 0; j < num_clusters; ++j)
            if (counts[j] > 0)
                for (h = 0; h < features; ++h)
                    centroids[j * features + h] = new_centroids[j * features + h] / counts[j];

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