# Fuzzy C-Means: CPU vs GPU (CUDA)

Implementazione del Fuzzy C-Means (FCM) in due varianti:
- **CPU**: single-thread, codice di riferimento (`fcm_cpu.cpp`)
- **GPU**: CUDA, kernel paralleli (`fcm_gpu.cu`)

su un dataset sintetico generato a runtime (blob gaussiani in D dimensioni).

## Struttura del progetto

```
dataset.h / dataset.cpp   generatore di dataset sintetico (blob gaussiani)
fcm_cpu.h / fcm_cpu.cpp    Fuzzy C-Means, versione CPU
fcm_gpu.cuh / fcm_gpu.cu   Fuzzy C-Means, versione CUDA
main.cpp                   orchestrazione: genera dati, esegue CPU+GPU, confronta
Makefile                    compilazione mista g++ / nvcc
```

## Schema di compilazione (g++ + nvcc)

| File             | Compilatore | Motivo |
|------------------|-------------|--------|
| `dataset.cpp`    | g++         | puro C++, nessuna dipendenza CUDA |
| `fcm_cpu.cpp`    | g++         | puro C++, nessuna dipendenza CUDA |
| `main.cpp`       | g++         | usa solo `fcm_gpu.cuh`, che espone una firma di funzione plain C++ (nessun tipo CUDA), quindi non serve nvcc |
| `fcm_gpu.cu`     | nvcc        | contiene i kernel CUDA |
| link finale      | nvcc        | necessario per risolvere i simboli della libreria runtime CUDA (cudart) |

```bash
make                  # compila con ARCH di default (sm_75)
make ARCH=sm_86        # specifica la compute capability della tua GPU
./fcm_fuzzy --N 1000000 --D 32 --K 20
```

Per scoprire la compute capability della tua GPU: `nvidia-smi --query-gpu=compute_cap --format=csv`

## Parametri da riga di comando

```
--N        numero di punti          (default 200000)
--D        numero di dimensioni     (default 32)
--K        numero di cluster        (default 10)
--m        fuzzifier                (default 2.0)
--max_iter numero massimo iterazioni (default 100)
--tol      soglia di convergenza    (default 1e-4)
--seed     seed RNG                 (default 42)
```

Esempi per testare lo speed-up CPU/GPU al variare di N e D (come discusso):

```bash
# variazione di N (D, K fissi)
for N in 100000 500000 1000000 5000000 10000000; do
    ./fcm_fuzzy --N $N --D 32 --K 30
done

# variazione di D (N, K fissi)
for D in 2 16 64 256 1024; do
    ./fcm_fuzzy --N 1000000 --D $D --K 30
done
```

## Note di design

- **Stessa inizializzazione per CPU e GPU**: la matrice di appartenenza
  iniziale `U_init` viene generata una sola volta sull'host e passata
  identica ai due run. Questo rende i risultati confrontabili: se le due
  implementazioni sono corrette, devono convergere a soluzioni con
  funzione obiettivo `J` molto simile (il programma stampa la differenza
  relativa a fine esecuzione).

- **Clamp delle distanze**: sia la versione CPU che quella GPU clampano la
  distanza al quadrato a `1e-10` per evitare divisioni per zero quando un
  punto coincide (quasi) con un centroide. È stato fatto nello stesso modo
  in entrambe le versioni proprio per renderle numericamente confrontabili.

- **Calcolo dei centroidi su GPU (`kernel_accumulate_centroids`)**: usa
  `atomicAdd` su accumulatori globali (un thread per punto, loop su K
  cluster e D dimensioni). È la versione più semplice e corretta da cui
  partire; se K*D è grande la contesa sugli atomic può diventare un collo
  di bottiglia. Un'estensione naturale (utile da menzionare in una
  relazione) è accumulare prima in shared memory per blocco e fare un solo
  atomicAdd per blocco invece che per thread.

- **Criterio di convergenza su GPU senza trasferire tutta la matrice U**:
  il kernel `kernel_max_diff` usa `atomicMax` su `unsigned int`,
  interpretando il bit-pattern IEEE-754 di un float non-negativo (un
  valore assoluto è sempre >= 0) come intero. Per float non-negativi
  l'ordinamento dei bit-pattern coincide con l'ordinamento dei valori,
  quindi il trucco è corretto e permette di calcolare il massimo
  scambiando solo 4 byte tra host e device per iterazione, invece di
  copiare l'intera matrice N×K.

- **Misurazione dei tempi**: la versione CPU usa
  `std::chrono::high_resolution_clock`; la versione GPU usa `cudaEvent`
  attorno al ciclo di iterazioni (kernel + memcpy device-device/host-device
  necessari per il check di convergenza), così il confronto è tra "tempo
  di calcolo effettivo" nei due casi, escludendo l'overhead di
  inizializzazione del contesto CUDA driver/runtime.

## Possibili estensioni per il progetto

- Aggiungere la riduzione in shared memory per `kernel_accumulate_centroids`
  e misurare l'impatto sullo speed-up.
- Confrontare con un kernel k-means classico (hard assignment) per mostrare
  la differenza di pattern di calcolo (denso/regolare in FCM vs argmin con
  branching nel k-means).
- Usare `nvprof`/`nsys`/Nsight Compute per profilare dove va il tempo GPU
  (es. se gli atomicAdd dominano rispetto al calcolo delle distanze).
