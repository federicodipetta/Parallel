/**
 * test_parser.cpp
 * ══════════════════════════════════════════════════════════════════════
 * Test e demo del KDD Cup 1999 parser
 *
 * Compile:
 *   g++ -O3 -std=c++17 -o test_parser test_parser.cpp
 *
 * Usage:
 *   ./test_parser kddcup.data           # file labeled
 *   ./test_parser kddcup.data.unlabeled # file unlabeled
 *   ./test_parser kddcup.data 100000    # leggi solo le prime 100k righe
 * ══════════════════════════════════════════════════════════════════════
 */

#include <iomanip>
#include <iostream>
#include <string>
#include "parser.cpp"
// ── Funzione che conta i valori unici in una colonna ─────────────────
void inspect_column(const DataMatrix& dm, int col_idx,
                    const std::string& col_name, int top_n = 5)
{
    std::unordered_map<float,int> freq;
    for (int i = 0; i < dm.N; ++i)
        freq[dm.data[i * dm.D + col_idx]]++;

    std::vector<std::pair<float,int>> sorted(freq.begin(), freq.end());
    std::sort(sorted.begin(), sorted.end(),
              [](auto& a, auto& b){ return b.second < a.second; });

    std::cout << "  [" << std::setw(2) << col_idx << "] "
              << std::left << std::setw(30) << col_name
              << " unique=" << sorted.size();
    if (sorted.size() <= (size_t)top_n) {
        std::cout << "  vals: ";
        for (auto& [v, c] : sorted)
            std::cout << v << "×" << c << " ";
    }
    std::cout << "\n";
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <kddcup.data> [max_rows] [--no-labels]\n";
        return 1;
    }

    std::string filepath  = argv[1];
    size_t      max_rows  = 0;
    bool        labeled   = true;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--no-labels") labeled = false;
        else try { max_rows = std::stoull(arg); } catch (...) {}
    }

    std::cout << "══════════════════════════════════════════\n";
    std::cout << " KDD Cup 1999 Parser — Demo\n";
    std::cout << "══════════════════════════════════════════\n\n";

    // ── 1. Parsing ────────────────────────────────────────────────────
    std::cout << "[1] Parsing file: " << filepath << "\n";
    std::cout << "    load_labels=" << (labeled ? "true" : "false")
              << "  max_rows=" << (max_rows ? std::to_string(max_rows) : "all")
              << "\n\n";

    KDDParser parser;
    parser.verbose = true;
    parser.load(filepath, labeled, max_rows);

    // ── 2. Distribuzione label ────────────────────────────────────────
    if (labeled) {
        parser.print_label_distribution();

        auto ul = parser.unique_labels();
        std::cout << "\nLabel uniche nel file (" << ul.size() << "):\n  ";
        for (auto& l : ul) std::cout << l << " ";
        std::cout << "\n\nSuggerimento K per clustering: "
                  << ul.size() << " (una per classe)\n";
    }

    // ── 3. Conversione in matrice ─────────────────────────────────────
    std::cout << "\n[2] Conversione in DataMatrix (normalize=true, log_scale=true)\n";
    DataMatrix dm = parser.to_matrix(/*normalize=*/true, /*log_scale=*/true);

    std::cout << "\nMatrice: " << dm.N << " righe × " << dm.D << " colonne\n";
    std::cout << "Normalizzata: " << (dm.normalized ? "sì" : "no") << "\n";
    std::cout << "Memoria: " << (dm.N * dm.D * sizeof(float)) / (1024*1024)
              << " MB\n";

    // ── 4. Ispezione prime colonne ────────────────────────────────────
    std::cout << "\n[3] Ispezione colonne (post-normalizzazione):\n";
    for (int c = 0; c < dm.D; ++c)
        inspect_column(dm, c, KDD_COL_NAMES[c], 4);

    // ── 5. Prime 3 righe per verifica visiva ──────────────────────────
    std::cout << "\n[4] Prime 3 righe (valori normalizzati):\n";
    int show = std::min(dm.N, 3);
    for (int i = 0; i < show; ++i) {
        std::cout << "  Row " << i;
        if (dm.label_ids)
            std::cout << "  label=" << dm.label_names[dm.label_ids[i]];
        std::cout << "\n  ";
        for (int c = 0; c < dm.D; ++c)
            std::cout << std::fixed << std::setprecision(3)
                      << dm.data[i * dm.D + c] << " ";
        std::cout << "\n";
    }

    // ── 6. Salva binario (per ricarica veloce senza re-parsing) ───────
    std::string bin_path = filepath + ".bin";
    std::cout << "\n[5] Salvataggio matrice binaria: " << bin_path << "\n";
    parser.save_binary(dm, bin_path);

    // ── 7. Test ricarica binario ───────────────────────────────────────
    std::cout << "\n[6] Test ricarica da binario:\n";
    DataMatrix dm2 = KDDParser::load_binary(bin_path, labeled);
    std::cout << "    Verifica N=" << dm2.N << " D=" << dm2.D << " — "
              << (dm2.N == dm.N && dm2.D == dm.D ? "OK" : "ERRORE") << "\n";

    std::cout << "\n══════════════════════════════════════════\n";
    std::cout << " Done. Usa dm.data (float[" << dm.N << "x" << dm.D
              << "]) con il tuo Mini-Batch K-Means.\n";
    std::cout << "══════════════════════════════════════════\n";

    return 0;
}
