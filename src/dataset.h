#ifndef DATASET_H
#define DATASET_H

#include <vector>

// Genera un dataset sintetico di N punti in D dimensioni, distribuiti
// attorno a K centri ("blob") posizionati casualmente in [-spread, spread]^D,
// con rumore gaussiano isotropo di deviazione standard cluster_std.
//
// data:        output, row-major, dimensione N*D  (data[i*D + d])
// true_labels: output, dimensione N, etichetta del blob di origine
//              (NON usata dal Fuzzy C-Means, serve solo come riferimento
//              per eventuali analisi di qualità del clustering)
void generate_synthetic_blobs(std::vector<float>& data,
                               std::vector<int>& true_labels,
                               int N, int D, int K,
                               float spread, float cluster_std,
                               unsigned int seed);

#endif // DATASET_H
