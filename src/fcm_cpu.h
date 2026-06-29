#ifndef FCM_CPU_H
#define FCM_CPU_H

// Esegue il Fuzzy C-Means su CPU (singolo thread, codice di riferimento).
//
// data:     N x D, row-major, dataset di input (read-only)
// N, D, K:  numero di punti, dimensioni, numero di cluster
// m:        fuzzifier (tipicamente 2.0)
// max_iter: numero massimo di iterazioni
// tol:      soglia di convergenza (max |U_new - U_old|)
// U_init:   N x K, matrice di appartenenza iniziale (read-only, NON modificata)
// U_out:    N x K, matrice di appartenenza finale (output)
// C_out:    K x D, centroidi finali (output)
// iters_out: numero di iterazioni effettivamente eseguite (output, opzionale)
//
// Ritorna il tempo di calcolo in millisecondi.
double fcm_cpu_run(const float* data, int N, int D, int K, float m,
                    int max_iter, float tol,
                    const float* U_init,
                    float* U_out, float* C_out, int* iters_out);

#endif // FCM_CPU_H
