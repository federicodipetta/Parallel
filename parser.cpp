/**
 * kdd_parser.hpp
 * ══════════════════════════════════════════════════════════════════════
 * Parser header-only per KDD Cup 1999 Network Intrusion Detection Dataset
 *
 * Formato: CSV senza header, 41 feature + 1 label opzionale
 *
 * Colonne (0-indexed):
 *   0       duration          int
 *   1       protocol_type     string  { tcp, udp, icmp }
 *   2       service           string  { http, ftp, smtp, ... ~70 valori }
 *   3       flag              string  { SF, S0, REJ, RSTO, ... ~11 valori }
 *   4       src_bytes         int
 *   5       dst_bytes         int
 *   6-11    land,wrong_fragment,urgent,hot,num_failed_logins,logged_in  int
 *   12-21   vari contatori    int
 *   22-30   rate features     float
 *   31-40   host rate features float
 *   41      label             string  (opzionale, es. "normal.", "neptune.")
 *
 * Variabili categoriche encode:
 *   protocol_type : tcp=0, udp=1, icmp=2
 *   service       : mappatura automatica in ordine di prima apparizione
 *   flag          : SF=0, S0=1, REJ=2, RSTO=3, ...
 *
 * Uso:
 *   KDDParser parser;
 *   parser.load("kddcup.data", true);   // true = cattura label
 *   DataMatrix dm = parser.to_matrix(); // float[N×D], D=41 o 42
 *
 * Compile:
 *   g++ -O3 -std=c++17 -o test test_parser.cpp
 * ══════════════════════════════════════════════════════════════════════
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────
// Strutture dati
// ─────────────────────────────────────────────────────────────────────

/** Una singola riga del dataset (valori ancora raw/tipizzati) */
struct KDDRecord {
    // Feature numeriche intere (col 0,4,5,6..21,31,32)
    int duration;
    int src_bytes;
    int dst_bytes;
    int land;
    int wrong_fragment;
    int urgent;
    int hot;
    int num_failed_logins;
    int logged_in;
    int num_compromised;
    int root_shell;
    int su_attempted;
    int num_root;
    int num_file_creations;
    int num_shells;
    int num_access_files;
    int num_outbound_cmds;
    int is_host_login;
    int is_guest_login;
    int count;
    int srv_count;
    int dst_host_count;
    int dst_host_srv_count;

    // Feature numeriche float (col 22..30, 33..40)
    float serror_rate;
    float srv_serror_rate;
    float rerror_rate;
    float srv_rerror_rate;
    float same_srv_rate;
    float diff_srv_rate;
    float srv_diff_host_rate;
    float dst_host_same_srv_rate;
    float dst_host_diff_srv_rate;
    float dst_host_same_src_port_rate;
    float dst_host_srv_diff_host_rate;
    float dst_host_serror_rate;
    float dst_host_srv_serror_rate;
    float dst_host_rerror_rate;
    float dst_host_srv_rerror_rate;

    // Feature categoriche (codificate come int dopo il parsing)
    int protocol_type;  // tcp=0, udp=1, icmp=2
    int service;        // indice nella mappa service
    int flag;           // indice nella mappa flag

    // Label (vuota se non catturata)
    std::string label;
    int         label_id;  // -1 se non catturata

    KDDRecord() : label_id(-1) {}
};

/** Matrice float pronta per il clustering */
struct  DataMatrix {
    float*      data;       // [N × D] row-major
    int32_t*    label_ids;  // [N]  — nullptr se load_labels=false
    int         N;
    int         D;          // 41 feature numeriche (categoriche one-hot o encoded)

    // Mappe per de-codificare
    std::vector<std::string> protocol_names;  // idx → nome
    std::vector<std::string> service_names;
    std::vector<std::string> flag_names;
    std::vector<std::string> label_names;     // idx → "normal.", "neptune." ...

    // Statistiche normalizzazione
    std::vector<float> col_mean;
    std::vector<float> col_std;
    bool normalized = false;

    DataMatrix() : data(nullptr), label_ids(nullptr), N(0), D(0) {}
    ~DataMatrix() { delete[] data; delete[] label_ids; }

    DataMatrix(const DataMatrix&) = delete;
    DataMatrix& operator=(const DataMatrix&) = delete;
    DataMatrix(DataMatrix&& o) noexcept
        : data(o.data), label_ids(o.label_ids), N(o.N), D(o.D),
          protocol_names(std::move(o.protocol_names)),
          service_names(std::move(o.service_names)),
          flag_names(std::move(o.flag_names)),
          label_names(std::move(o.label_names)),
          col_mean(std::move(o.col_mean)),
          col_std(std::move(o.col_std)),
          normalized(o.normalized)
    { o.data = nullptr; o.label_ids = nullptr; }
};

// ─────────────────────────────────────────────────────────────────────
// Indici colonne (per riferimento rapido)
// ─────────────────────────────────────────────────────────────────────
static const char* KDD_COL_NAMES[41] = {
    /*  0 */ "duration",
    /*  1 */ "protocol_type",
    /*  2 */ "service",
    /*  3 */ "flag",
    /*  4 */ "src_bytes",
    /*  5 */ "dst_bytes",
    /*  6 */ "land",
    /*  7 */ "wrong_fragment",
    /*  8 */ "urgent",
    /*  9 */ "hot",
    /* 10 */ "num_failed_logins",
    /* 11 */ "logged_in",
    /* 12 */ "num_compromised",
    /* 13 */ "root_shell",
    /* 14 */ "su_attempted",
    /* 15 */ "num_root",
    /* 16 */ "num_file_creations",
    /* 17 */ "num_shells",
    /* 18 */ "num_access_files",
    /* 19 */ "num_outbound_cmds",
    /* 20 */ "is_host_login",
    /* 21 */ "is_guest_login",
    /* 22 */ "count",
    /* 23 */ "srv_count",
    /* 24 */ "serror_rate",
    /* 25 */ "srv_serror_rate",
    /* 26 */ "rerror_rate",
    /* 27 */ "srv_rerror_rate",
    /* 28 */ "same_srv_rate",
    /* 29 */ "diff_srv_rate",
    /* 30 */ "srv_diff_host_rate",
    /* 31 */ "dst_host_count",
    /* 32 */ "dst_host_srv_count",
    /* 33 */ "dst_host_same_srv_rate",
    /* 34 */ "dst_host_diff_srv_rate",
    /* 35 */ "dst_host_same_src_port_rate",
    /* 36 */ "dst_host_srv_diff_host_rate",
    /* 37 */ "dst_host_serror_rate",
    /* 38 */ "dst_host_srv_serror_rate",
    /* 39 */ "dst_host_rerror_rate",
    /* 40 */ "dst_host_srv_rerror_rate",
};

// ─────────────────────────────────────────────────────────────────────
// Classe parser
// ─────────────────────────────────────────────────────────────────────
class KDDParser {
public:
    // ── Opzioni ───────────────────────────────────────────────────────
    bool verbose      = true;    // stampa progresso
    bool load_labels  = true;    // cattura colonna label (col 41)
    char delimiter    = ',';

    // ── Dati grezzi letti ─────────────────────────────────────────────
    std::vector<KDDRecord> records;

    // ── Mappe categoriche (costruite durante il parsing) ──────────────
    std::unordered_map<std::string, int> protocol_map;
    std::unordered_map<std::string, int> service_map;
    std::unordered_map<std::string, int> flag_map;
    std::unordered_map<std::string, int> label_map;

    std::vector<std::string> protocol_names;
    std::vector<std::string> service_names;
    std::vector<std::string> flag_names;
    std::vector<std::string> label_names;

    // ── Statistiche parse ─────────────────────────────────────────────
    size_t lines_read    = 0;
    size_t lines_skipped = 0;
    double parse_ms      = 0.0;

    // ─────────────────────────────────────────────────────────────────
    // Costruttore
    // ─────────────────────────────────────────────────────────────────
    KDDParser() {
        // Pre-popola le mappe con i valori noti del KDD Cup '99
        // così l'encoding è stabile anche su file parziali

        // protocol_type
        _register(protocol_map, protocol_names, "tcp");    // 0
        _register(protocol_map, protocol_names, "udp");    // 1
        _register(protocol_map, protocol_names, "icmp");   // 2

        // flag
        _register(flag_map, flag_names, "SF");      // 0  — normal
        _register(flag_map, flag_names, "S0");      // 1
        _register(flag_map, flag_names, "REJ");     // 2
        _register(flag_map, flag_names, "RSTO");    // 3
        _register(flag_map, flag_names, "RSTR");    // 4
        _register(flag_map, flag_names, "SH");      // 5
        _register(flag_map, flag_names, "S1");      // 6
        _register(flag_map, flag_names, "S2");      // 7
        _register(flag_map, flag_names, "S3");      // 8
        _register(flag_map, flag_names, "OTH");     // 9
        _register(flag_map, flag_names, "RSTOS0");  // 10

        // service: i più comuni — nuovi vengono aggiunti dinamicamente
        for (const char* s : {
            "http","smtp","finger","domain_u","auth","telnet","ftp",
            "eco_i","ntp_u","ecr_i","other","private","pop_3","ftp_data",
            "rpc","kshell","klogin","urp_i","X11","Z39_50","time","nntp",
            "shell","login","imap4","nnsp","http_443","exec","printer",
            "efs","courier","uucp","whois","netbios_ssn","netbios_dgm",
            "netbios_ns","pm_dump","remote_job","ctf","netstat","ircx",
            "iso_tsap","ldap","sunrpc","pop_2","link","systat","supdup",
            "discard","daytime","ssh","hostnames","csnet_ns","uucp_path",
            "vmnet","bgp","IRC","echo","name","domain","mtp","gopher",
            "tim_i","tftp_u","sql_net","urh_i","http_8001","aol",
            "http_2784","harvest"
        }) {
            _register(service_map, service_names, s);
        }

        // label — le 23 classi del KDD Cup '99
        for (const char* l : {
            "normal.",
            "neptune.","smurf.","pod.","teardrop.","back.","land.",
            "ipsweep.","portsweep.","nmap.","satan.",
            "guess_passwd.","ftp_write.","imap.","phf.","multihop.",
            "warezmaster.","warezclient.","spy.","rootkit.","buffer_overflow.",
            "loadmodule.","perl.","sendmail."
        }) {
            _register(label_map, label_names, l);
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // load() — legge il file e popola records[]
    // ─────────────────────────────────────────────────────────────────
    /**
     * @param filepath     percorso al file CSV
     * @param with_labels  true = il file ha la colonna 41 con la label
     *                     false = file unlabeled (41 colonne totali)
     * @param max_rows     0 = leggi tutto; >0 = leggi solo i primi N
     */
    void load(const std::string& filepath,
              bool with_labels = true,
              size_t max_rows  = 0)
    {
        load_labels = with_labels;

        std::ifstream f(filepath);
        if (!f.is_open())
            throw std::runtime_error("[KDDParser] Impossibile aprire: " + filepath);

        auto t0 = _now();
        records.clear();
        records.reserve(500'000);  // KDD full ha ~5M righe

        std::string line;
        lines_read = lines_skipped = 0;

        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            // rimuovi '\r' (file Windows)
            if (!line.empty() && line.back() == '\r') line.pop_back();

            ++lines_read;
            if (max_rows > 0 && records.size() >= max_rows) break;

            KDDRecord rec;
            if (!_parse_line(line, rec)) {
                ++lines_skipped;
                if (verbose && lines_skipped <= 5)
                    std::cerr << "[KDDParser] Riga " << lines_read
                              << " malformata, skip: " << line.substr(0,60) << "\n";
                continue;
            }
            records.push_back(std::move(rec));
        }

        parse_ms = _elapsed(t0);

        if (verbose) {
            std::cout << "[KDDParser] Lette "   << records.size() << " righe"
                      << "  (skip=" << lines_skipped << ")"
                      << "  in " << parse_ms << " ms\n";
            _print_stats();
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // to_matrix() — converte records in float[N×D]
    // ─────────────────────────────────────────────────────────────────
    /**
     * Ordine colonne nella matrice output (D=41):
     *   [0]     duration
     *   [1]     protocol_type  (encoded 0/1/2)
     *   [2]     service        (encoded 0..N_service)
     *   [3]     flag           (encoded 0..10)
     *   [4]     src_bytes
     *   [5]     dst_bytes
     *   [6-21]  feature intere rimanenti
     *   [22-40] feature float
     *
     * @param normalize   applica z-score per colonna (consigliato per K-Means)
     * @param log_scale   applica log1p alle colonne con range ampio (bytes, count)
     *                    prima della normalizzazione
     */
    DataMatrix to_matrix(bool normalize  = true,
                         bool log_scale  = true) const
    {
        const int N = (int)records.size();
        const int D = 41;
        assert(N > 0 && "Nessun record: chiama load() prima");

        DataMatrix dm;
        dm.N = N;
        dm.D = D;
        dm.data      = new float[N * D];
        dm.label_ids = load_labels ? new int32_t[N] : nullptr;
        dm.protocol_names = protocol_names;
        dm.service_names  = service_names;
        dm.flag_names     = flag_names;
        dm.label_names    = label_names;

        // ── Converti ogni record in riga float ────────────────────────
        for (int i = 0; i < N; ++i) {
            const KDDRecord& r = records[i];
            float* row = dm.data + i * D;

            row[ 0] = (float)r.duration;
            row[ 1] = (float)r.protocol_type;
            row[ 2] = (float)r.service;
            row[ 3] = (float)r.flag;
            row[ 4] = (float)r.src_bytes;
            row[ 5] = (float)r.dst_bytes;
            row[ 6] = (float)r.land;
            row[ 7] = (float)r.wrong_fragment;
            row[ 8] = (float)r.urgent;
            row[ 9] = (float)r.hot;
            row[10] = (float)r.num_failed_logins;
            row[11] = (float)r.logged_in;
            row[12] = (float)r.num_compromised;
            row[13] = (float)r.root_shell;
            row[14] = (float)r.su_attempted;
            row[15] = (float)r.num_root;
            row[16] = (float)r.num_file_creations;
            row[17] = (float)r.num_shells;
            row[18] = (float)r.num_access_files;
            row[19] = (float)r.num_outbound_cmds;
            row[20] = (float)r.is_host_login;
            row[21] = (float)r.is_guest_login;
            row[22] = (float)r.count;
            row[23] = (float)r.srv_count;
            row[24] = r.serror_rate;
            row[25] = r.srv_serror_rate;
            row[26] = r.rerror_rate;
            row[27] = r.srv_rerror_rate;
            row[28] = r.same_srv_rate;
            row[29] = r.diff_srv_rate;
            row[30] = r.srv_diff_host_rate;
            row[31] = (float)r.dst_host_count;
            row[32] = (float)r.dst_host_srv_count;
            row[33] = r.dst_host_same_srv_rate;
            row[34] = r.dst_host_diff_srv_rate;
            row[35] = r.dst_host_same_src_port_rate;
            row[36] = r.dst_host_srv_diff_host_rate;
            row[37] = r.dst_host_serror_rate;
            row[38] = r.dst_host_srv_serror_rate;
            row[39] = r.dst_host_rerror_rate;
            row[40] = r.dst_host_srv_rerror_rate;

            if (load_labels && dm.label_ids)
                dm.label_ids[i] = r.label_id;
        }

        // ── Log1p su colonne con range ampio ──────────────────────────
        // src_bytes, dst_bytes, num_compromised, num_root, count, srv_count,
        // dst_host_count, dst_host_srv_count possono arrivare a milioni
        if (log_scale) {
            static const int log_cols[] = {
                0,   // duration
                4,   // src_bytes
                5,   // dst_bytes
                9,   // hot
                12,  // num_compromised
                15,  // num_root
                22,  // count
                23,  // srv_count
                31,  // dst_host_count
                32,  // dst_host_srv_count
                -1
            };
            for (int ci = 0; log_cols[ci] >= 0; ++ci) {
                int c = log_cols[ci];
                for (int i = 0; i < N; ++i)
                    dm.data[i * D + c] = std::log1p(dm.data[i * D + c]);
            }
            if (verbose)
                std::cout << "[KDDParser] log1p applicato alle colonne ad alto range\n";
        }

        // ── Z-score normalizzazione ───────────────────────────────────
        if (normalize) {
            dm.col_mean.resize(D, 0.0f);
            dm.col_std.resize(D, 1.0f);

            // Calcola medie
            for (int c = 0; c < D; ++c) {
                double s = 0.0;
                for (int i = 0; i < N; ++i) s += dm.data[i * D + c];
                dm.col_mean[c] = (float)(s / N);
            }
            // Calcola std
            for (int c = 0; c < D; ++c) {
                double s2 = 0.0;
                float  m  = dm.col_mean[c];
                for (int i = 0; i < N; ++i) {
                    float diff = dm.data[i * D + c] - m;
                    s2 += diff * diff;
                }
                dm.col_std[c] = (float)std::sqrt(s2 / N + 1e-8);
            }
            // Applica
            for (int i = 0; i < N; ++i)
                for (int c = 0; c < D; ++c)
                    dm.data[i * D + c] =
                        (dm.data[i * D + c] - dm.col_mean[c]) / dm.col_std[c];

            dm.normalized = true;
            if (verbose)
                std::cout << "[KDDParser] Normalizzazione z-score applicata\n";
        }

        return dm;
    }

    // ─────────────────────────────────────────────────────────────────
    // Utilità pubbliche
    // ─────────────────────────────────────────────────────────────────

    /** Distribuzione delle label nel dataset caricato */
    void print_label_distribution() const {
        if (!load_labels) { std::cout << "Labels non caricate\n"; return; }
        std::unordered_map<int,int> cnt;
        for (auto& r : records) cnt[r.label_id]++;
        std::cout << "\nDistribuzione label (" << records.size() << " righe):\n";
        // ordina per frequenza
        std::vector<std::pair<int,int>> sorted(cnt.begin(), cnt.end());
        std::sort(sorted.begin(), sorted.end(),
                  [](auto& a, auto& b){ return b.second < a.second; });
        for (auto& [id, n] : sorted) {
            std::string name = (id >= 0 && id < (int)label_names.size())
                               ? label_names[id] : "?";
            std::cout << "  " << name
                      << std::string(25 - name.size(), ' ')
                      << n << "\n";
        }
    }

    /** Ritorna le label uniche presenti (per scegliere K nel clustering) */
    std::vector<std::string> unique_labels() const {
        std::vector<bool> seen(label_names.size(), false);
        for (auto& r : records)
            if (r.label_id >= 0 && r.label_id < (int)seen.size())
                seen[r.label_id] = true;
        std::vector<std::string> out;
        for (size_t i = 0; i < seen.size(); ++i)
            if (seen[i]) out.push_back(label_names[i]);
        return out;
    }

    /** Salva la matrice su file binario (ricaricabile senza re-parsing) */
    void save_binary(const DataMatrix& dm, const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(&dm.N), sizeof(int));
        f.write(reinterpret_cast<const char*>(&dm.D), sizeof(int));
        f.write(reinterpret_cast<const char*>(dm.data),
                (size_t)dm.N * dm.D * sizeof(float));
        if (dm.label_ids)
            f.write(reinterpret_cast<const char*>(dm.label_ids),
                    (size_t)dm.N * sizeof(int32_t));
        std::cout << "[KDDParser] Matrice salvata: " << path
                  << "  (" << dm.N << "×" << dm.D << ")\n";
    }

    /** Carica matrice da file binario (bypass del CSV parsing) */
    static DataMatrix load_binary(const std::string& path,
                                  bool has_labels = true) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open())
            throw std::runtime_error("Impossibile aprire: " + path);
        DataMatrix dm;
        f.read(reinterpret_cast<char*>(&dm.N), sizeof(int));
        f.read(reinterpret_cast<char*>(&dm.D), sizeof(int));
        dm.data = new float[(size_t)dm.N * dm.D];
        f.read(reinterpret_cast<char*>(dm.data),
               (size_t)dm.N * dm.D * sizeof(float));
        if (has_labels) {
            dm.label_ids = new int32_t[dm.N];
            f.read(reinterpret_cast<char*>(dm.label_ids),
                   (size_t)dm.N * sizeof(int32_t));
        }
        std::cout << "[KDDParser] Matrice caricata: " << path
                  << "  (" << dm.N << "x" << dm.D << ")\n";
        return dm;
    }

// ─────────────────────────────────────────────────────────────────────
private:
// ─────────────────────────────────────────────────────────────────────

    // ── Parsing di una singola riga ───────────────────────────────────
    bool _parse_line(const std::string& line, KDDRecord& rec) {
        // Split veloce senza allocare un vettore di stringhe:
        // usa string_view-like con puntatori alla riga originale
        const char* p   = line.c_str();
        const char* end = p + line.size();

        // col[i] = {start, len} all'interno di 'line'
        // Il KDD ha esattamente 41 o 42 colonne
        const char* fields[43];
        int         flens[43];
        int         ncols = 0;

        const char* fs = p;
        while (p <= end && ncols < 43) {
            if (p == end || *p == delimiter) {
                fields[ncols] = fs;
                flens[ncols]  = (int)(p - fs);
                ++ncols;
                fs = p + 1;
            }
            ++p;
        }

        // Deve avere 41 (unlabeled) o 42 (labeled) colonne
        if (ncols < 41) return false;

        // Helper lambda: leggi int dal campo i
        auto get_int = [&](int i) -> int {
            return std::stoi(std::string(fields[i], flens[i]));
        };
        // Helper lambda: leggi float dal campo i
        auto get_flt = [&](int i) -> float {
            return std::stof(std::string(fields[i], flens[i]));
        };
        // Helper lambda: leggi string dal campo i
        auto get_str = [&](int i) -> std::string {
            return std::string(fields[i], flens[i]);
        };

        try {
            rec.duration          = get_int(0);
            rec.protocol_type     = _encode(get_str(1), protocol_map, protocol_names);
            rec.service           = _encode(get_str(2), service_map,  service_names);
            rec.flag              = _encode(get_str(3), flag_map,     flag_names);
            rec.src_bytes         = get_int(4);
            rec.dst_bytes         = get_int(5);
            rec.land              = get_int(6);
            rec.wrong_fragment    = get_int(7);
            rec.urgent            = get_int(8);
            rec.hot               = get_int(9);
            rec.num_failed_logins = get_int(10);
            rec.logged_in         = get_int(11);
            rec.num_compromised   = get_int(12);
            rec.root_shell        = get_int(13);
            rec.su_attempted      = get_int(14);
            rec.num_root          = get_int(15);
            rec.num_file_creations= get_int(16);
            rec.num_shells        = get_int(17);
            rec.num_access_files  = get_int(18);
            rec.num_outbound_cmds = get_int(19);
            rec.is_host_login     = get_int(20);
            rec.is_guest_login    = get_int(21);
            rec.count             = get_int(22);
            rec.srv_count         = get_int(23);
            rec.serror_rate       = get_flt(24);
            rec.srv_serror_rate   = get_flt(25);
            rec.rerror_rate       = get_flt(26);
            rec.srv_rerror_rate   = get_flt(27);
            rec.same_srv_rate     = get_flt(28);
            rec.diff_srv_rate     = get_flt(29);
            rec.srv_diff_host_rate= get_flt(30);
            rec.dst_host_count    = get_int(31);
            rec.dst_host_srv_count= get_int(32);
            rec.dst_host_same_srv_rate       = get_flt(33);
            rec.dst_host_diff_srv_rate       = get_flt(34);
            rec.dst_host_same_src_port_rate  = get_flt(35);
            rec.dst_host_srv_diff_host_rate  = get_flt(36);
            rec.dst_host_serror_rate         = get_flt(37);
            rec.dst_host_srv_serror_rate     = get_flt(38);
            rec.dst_host_rerror_rate         = get_flt(39);
            rec.dst_host_srv_rerror_rate     = get_flt(40);

            // Label (colonna 41) — solo se presente E richiesta
            if (load_labels && ncols >= 42) {
                std::string lbl = get_str(41);
                // rimuovi punto finale se mancante (es. "normal" → "normal.")
                if (!lbl.empty() && lbl.back() != '.') lbl += '.';
                rec.label    = lbl;
                rec.label_id = _encode(lbl, label_map, label_names);
            } else {
                rec.label_id = -1;
            }
        } catch (...) {
            return false;
        }

        return true;
    }

    // ── Encoding categorico ───────────────────────────────────────────
    static int _encode(const std::string& val,
                       std::unordered_map<std::string,int>& map,
                       std::vector<std::string>& names)
    {
        auto it = map.find(val);
        if (it != map.end()) return it->second;
        int id = (int)names.size();
        map[val] = id;
        names.push_back(val);
        return id;
    }

    static void _register(std::unordered_map<std::string,int>& map,
                           std::vector<std::string>& names,
                           const std::string& val)
    {
        if (map.find(val) == map.end()) {
            map[val] = (int)names.size();
            names.push_back(val);
        }
    }

    // ── Statistiche post-load ─────────────────────────────────────────
    void _print_stats() const {
        std::cout << "[KDDParser] Protocol types : " << protocol_names.size() << "\n";
        std::cout << "[KDDParser] Services       : " << service_names.size()  << "\n";
        std::cout << "[KDDParser] Flags          : " << flag_names.size()     << "\n";
        if (load_labels)
            std::cout << "[KDDParser] Label classes  : " << label_names.size() << "\n";
    }

    // ── Timer ─────────────────────────────────────────────────────────
    using TP = std::chrono::time_point<std::chrono::high_resolution_clock>;
    static TP _now() { return std::chrono::high_resolution_clock::now(); }
    static double _elapsed(TP t0) {
        return std::chrono::duration<double,std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
    }
};