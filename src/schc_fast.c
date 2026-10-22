/*
 * Fast CPU implementation of SCHC (open boundaries), for the size scans.
 *
 * Follows the rules of src/simulation.py (the JAX reference) with work proportional to
 * the components involved instead of the whole grid:
 *   - two cells drawn independently and uniformly from the active cells (with replacement);
 *   - 8-connected components; score = "xor" or "mix" hash of the translation-normalized
 *     component, compared as float32 exactly as in JAX;
 *   - winner rule "original" (higher score wins, ties uniform) or "legacy" (the published
 *     JAX rule: a strictly higher second contestant wins only with probability 1/2);
 *   - the winner's bounding box plus a one-cell margin is copied onto the loser's
 *     bounding-box center (rint, half to even), reading the pre-contest configuration,
 *     clipped at the grid edges, with per-cell death / mutation (also on empty cells);
 *   - clock: elapsed += max(|winner| / N_active, 1e-6) until elapsed >= 1 (float32).
 * Random streams differ from JAX (xoshiro256**), so runs agree in distribution, not bit for bit.
 * Unlike the JAX scatter, a copy onto cell (0,0) is not lost.
 *
 * Build: gcc -O3 -march=native -shared -fPIC -o src/_schc_fast.so src/schc_fast.c -lm
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- RNG (xoshiro256**) */
typedef struct { uint64_t s[4]; } rng_t;

static inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
static inline uint64_t rng_next(rng_t *r) {
    uint64_t *s = r->s, result = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
    return result;
}
static void rng_seed(rng_t *r, uint64_t seed) {
    for (int i = 0; i < 4; i++) {           /* splitmix64 */
        uint64_t z = (seed += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
}
static inline double rng_uniform(rng_t *r) { return (rng_next(r) >> 11) * 0x1.0p-53; }
static inline uint32_t rng_below(rng_t *r, uint32_t n) { return (uint32_t)(((unsigned __int128)rng_next(r) * n) >> 64); }

/* ---------------------------------------------------------------- hashes (as in JAX) */
static inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16; h *= 0x85EBCA6BU; h ^= h >> 13; h *= 0xC2B2AE35U; h ^= h >> 16; return h;
}
/* cells: flat indices (x*L+y); returns uint32 hash; score 0 = xor, 1 = mix */
static uint32_t component_hash(const int32_t *config, int L, const int32_t *cells, int n, int score, int seed) {
    int minx = L, miny = L;
    for (int i = 0; i < n; i++) { int x = cells[i] / L, y = cells[i] % L; if (x < minx) minx = x; if (y < miny) miny = y; }
    if (score == 0) {
        uint32_t acc = 0;
        for (int i = 0; i < n; i++) {
            uint32_t rx = (uint32_t)(cells[i] / L - minx), ry = (uint32_t)(cells[i] % L - miny);
            acc ^= (rx * 0x9E3779B9U) ^ (ry * 0x85EBCA6BU) ^ (uint32_t)config[cells[i]];
        }
        return (acc ^ 0x45D9F3BU) * 0x27D4EB2DU;
    }
    uint32_t c0 = (uint32_t)((0x9E3779B9ULL + 0x632BE5ABULL * (uint64_t)seed) & 0xFFFFFFFFULL);
    uint32_t c1 = (uint32_t)((0x27D4EB2FULL * (uint64_t)(seed + 1)) & 0xFFFFFFFFULL);
    uint32_t total = 0;
    for (int i = 0; i < n; i++) {
        uint32_t rx = (uint32_t)(cells[i] / L - minx), ry = (uint32_t)(cells[i] % L - miny);
        uint32_t h = fmix32((uint32_t)config[cells[i]] + c0);
        h = fmix32(h ^ (rx * 0x85EBCA6BU + 0x7F4A7C15U));
        h = fmix32(h ^ (ry * 0xC2B2AE35U + 0x165667B1U));
        total += h;
    }
    return fmix32(total ^ c1);
}
static inline float hash_to_fitness(uint32_t h) { return (float)h / 4294967296.0f; } /* float32(UINT32_MAX) == 2^32 */

/* ---------------------------------------------------------------- world state */
typedef struct {
    int L, k, score, winner_rule;
    double mu, death;
    int32_t *config;      /* L*L, 0 = empty */
    int32_t *active;      /* list of active flat indices */
    int32_t *pos;         /* position of cell in active list, -1 if empty */
    int32_t n_active;
    int32_t *stamp; int32_t cur_stamp;   /* visited marks for flood fill */
    int32_t *comp1, *comp2, *queue;      /* scratch (L*L each) */
    int32_t *blockbuf;                   /* scratch ((L+2)^2) */
    rng_t rng;
} world_t;

static void set_cell(world_t *w, int idx, int32_t v) {
    int32_t old = w->config[idx];
    w->config[idx] = v;
    if (old == 0 && v != 0) { w->pos[idx] = w->n_active; w->active[w->n_active++] = idx; }
    else if (old != 0 && v == 0) {
        int p = w->pos[idx], last = w->active[--w->n_active];
        w->active[p] = last; w->pos[last] = p; w->pos[idx] = -1;
    }
}

/* 8-connected flood fill from start; returns size, cells in out */
static int flood(world_t *w, int start, int32_t *out) {
    int L = w->L, head = 0, tail = 0;
    if (++w->cur_stamp == INT32_MAX) { memset(w->stamp, 0, sizeof(int32_t) * L * L); w->cur_stamp = 1; }
    int32_t st = w->cur_stamp;
    w->stamp[start] = st; w->queue[tail++] = start;
    while (head < tail) {
        int c = w->queue[head++], x = c / L, y = c % L;
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx; if (nx < 0 || nx >= L) continue;
            for (int dy = -1; dy <= 1; dy++) {
                int ny = y + dy; if (ny < 0 || ny >= L) continue;
                int j = nx * L + ny;
                if (w->config[j] != 0 && w->stamp[j] != st) { w->stamp[j] = st; w->queue[tail++] = j; }
            }
        }
    }
    memcpy(out, w->queue, sizeof(int32_t) * tail);
    return tail;
}

static void bbox(int L, const int32_t *cells, int n, int *mn, int *mx) {
    mn[0] = mn[1] = L; mx[0] = mx[1] = -1;
    for (int i = 0; i < n; i++) {
        int x = cells[i] / L, y = cells[i] % L;
        if (x < mn[0]) mn[0] = x; if (y < mn[1]) mn[1] = y;
        if (x > mx[0]) mx[0] = x; if (y > mx[1]) mx[1] = y;
    }
}

static inline int32_t mutate(world_t *w, int32_t v) {
    double d = rng_uniform(&w->rng), m = rng_uniform(&w->rng);
    if (d < w->death) return 0;
    if (m < w->mu) return 1 + (int32_t)rng_below(&w->rng, (uint32_t)w->k);
    return v;
}

/* copy winner box (wmn..wmx, + margin) onto the loser's box center */
static void rewrite(world_t *w, const int *wmn, const int *wmx, const int *lmn, const int *lmx) {
    int L = w->L;
    int wc[2], lc[2], start[2], shape[2];
    for (int a = 0; a < 2; a++) {
        wc[a] = (int)nearbyint((wmn[a] + wmx[a]) / 2.0);   /* half to even, as jnp.rint */
        lc[a] = (int)nearbyint((lmn[a] + lmx[a]) / 2.0);
        start[a] = lc[a] + wmn[a] - wc[a] - 1;
        shape[a] = wmx[a] - wmn[a] + 3;
    }
    int bmin0 = wmn[0] - 1, bmin1 = wmn[1] - 1;
    /* read the source block from the pre-contest configuration first */
    for (int ox = 0; ox < shape[0]; ox++)
        for (int oy = 0; oy < shape[1]; oy++) {
            int gx = bmin0 + ox, gy = bmin1 + oy;
            w->blockbuf[ox * shape[1] + oy] = (gx >= 0 && gx < L && gy >= 0 && gy < L) ? w->config[gx * L + gy] : -1;
        }
    for (int ox = 0; ox < shape[0]; ox++)
        for (int oy = 0; oy < shape[1]; oy++) {
            int32_t src = w->blockbuf[ox * shape[1] + oy];
            int tx = start[0] + ox, ty = start[1] + oy;
            if (src < 0 || tx < 0 || tx >= L || ty < 0 || ty >= L) continue;
            set_cell(w, tx * L + ty, mutate(w, src));
        }
}

/* one contest; returns winner_prop (float32 semantics) */
static float contest(world_t *w, int forced, int s1, int s2, int coin) {
    if (w->n_active == 0) return 1.0f;
    int L = w->L, N = w->n_active;
    int start1 = forced ? s1 : w->active[rng_below(&w->rng, (uint32_t)N)];
    int start2 = forced ? s2 : w->active[rng_below(&w->rng, (uint32_t)N)];
    int n1 = flood(w, start1, w->comp1);
    int n2 = flood(w, start2, w->comp2);
    uint32_t h1 = component_hash(w->config, L, w->comp1, n1, w->score, 0);
    uint32_t h2 = component_hash(w->config, L, w->comp2, n2, w->score, 0);
    float f1 = hash_to_fitness(h1), f2 = hash_to_fitness(h2);
    int c = forced ? coin : (int)rng_below(&w->rng, 2);
    int winner;
    if (w->winner_rule == 1) winner = f1 > f2 ? 0 : (f2 > f1 ? 1 : c);
    else winner = f1 > f2 ? 0 : c;          /* legacy: f2 > f1 or tie -> uniform */
    int mn1[2], mx1[2], mn2[2], mx2[2];
    bbox(L, w->comp1, n1, mn1, mx1);
    bbox(L, w->comp2, n2, mn2, mx2);
    float prop = (float)(winner == 0 ? n1 : n2) / (float)(N > 1 ? N : 1);
    if (winner == 0) rewrite(w, mn1, mx1, mn2, mx2); else rewrite(w, mn2, mx2, mn1, mx1);
    return prop;
}

static void onestep(world_t *w) {
    float elapsed = 0.0f;
    while (elapsed < 1.0f) {
        float p = contest(w, 0, 0, 0, 0);
        elapsed += p > 1e-6f ? p : 1e-6f;
    }
}

/* ---------------------------------------------------------------- 64-bit hash set */
typedef struct { uint64_t *keys; size_t cap, n; } set64_t;
static void set_init(set64_t *s) { s->cap = 1 << 16; s->n = 0; s->keys = calloc(s->cap, sizeof(uint64_t)); }
static void set_insert(set64_t *s, uint64_t key);
static void set_grow(set64_t *s) {
    uint64_t *old = s->keys; size_t oc = s->cap;
    s->cap *= 2; s->n = 0; s->keys = calloc(s->cap, sizeof(uint64_t));
    for (size_t i = 0; i < oc; i++) if (old[i]) set_insert(s, old[i]);
    free(old);
}
static void set_insert(set64_t *s, uint64_t key) {
    if (key == 0) key = 1;                    /* 0 marks empty slots */
    if (2 * (s->n + 1) > s->cap) set_grow(s);
    size_t i = (size_t)(key * 0x9E3779B97F4A7C15ULL) & (s->cap - 1);
    while (s->keys[i]) { if (s->keys[i] == key) return; i = (i + 1) & (s->cap - 1); }
    s->keys[i] = key; s->n++;
}

/* ---------------------------------------------------------------- metrics */
/* row: max_fitness, mean_fitness, max_size, mean_size, n_active, n_components */
static void metrics(world_t *w, double *row, uint8_t *types_seen, set64_t *patterns) {
    int L = w->L, ncomp = 0, maxs = 0;
    double sumf = 0.0, maxf = 0.0;
    if (++w->cur_stamp == INT32_MAX) { memset(w->stamp, 0, sizeof(int32_t) * L * L); w->cur_stamp = 1; }
    int32_t st = w->cur_stamp;
    for (int i = 0; i < w->n_active; i++) types_seen[w->config[w->active[i]]] = 1;
    for (int idx = 0; idx < L * L; idx++) {
        if (w->config[idx] == 0 || w->stamp[idx] == st) continue;
        /* BFS with the shared stamp so that cells are visited once per sample */
        int head = 0, tail = 0; w->stamp[idx] = st; w->queue[tail++] = idx;
        while (head < tail) {
            int c = w->queue[head++], x = c / L, y = c % L;
            for (int dx = -1; dx <= 1; dx++) { int nx = x + dx; if (nx < 0 || nx >= L) continue;
                for (int dy = -1; dy <= 1; dy++) { int ny = y + dy; if (ny < 0 || ny >= L) continue;
                    int j = nx * L + ny;
                    if (w->config[j] != 0 && w->stamp[j] != st) { w->stamp[j] = st; w->queue[tail++] = j; } } }
        }
        ncomp++; if (tail > maxs) maxs = tail;
        double f = hash_to_fitness(component_hash(w->config, L, w->queue, tail, 1, 0));   /* "mix" score, as src/metrics.py */
        sumf += f; if (f > maxf) maxf = f;
        uint64_t id = ((uint64_t)component_hash(w->config, L, w->queue, tail, 1, 101) << 32)
                    | component_hash(w->config, L, w->queue, tail, 1, 202);
        set_insert(patterns, id);
    }
    row[0] = maxf; row[1] = ncomp ? sumf / ncomp : 0.0; row[2] = maxs;
    row[3] = ncomp ? (double)w->n_active / ncomp : 0.0; row[4] = w->n_active; row[5] = ncomp;
}

/* ---------------------------------------------------------------- API */
static world_t *world_new(int L, int k, double mu, double death, int score, int winner_rule, uint64_t seed) {
    world_t *w = calloc(1, sizeof(world_t));
    w->L = L; w->k = k; w->mu = mu; w->death = death; w->score = score; w->winner_rule = winner_rule;
    size_t n = (size_t)L * L;
    w->config = calloc(n, sizeof(int32_t)); w->active = calloc(n, sizeof(int32_t));
    w->pos = malloc(n * sizeof(int32_t)); for (size_t i = 0; i < n; i++) w->pos[i] = -1;
    w->stamp = calloc(n, sizeof(int32_t)); w->comp1 = calloc(n, sizeof(int32_t));
    w->comp2 = calloc(n, sizeof(int32_t)); w->queue = calloc(n, sizeof(int32_t));
    w->blockbuf = calloc((size_t)(L + 2) * (L + 2), sizeof(int32_t));
    rng_seed(&w->rng, seed);
    return w;
}
static void world_free(world_t *w) {
    free(w->config); free(w->active); free(w->pos); free(w->stamp); free(w->comp1);
    free(w->comp2); free(w->queue); free(w->blockbuf); free(w);
}

/*
 * Run one simulation. Rows (9 doubles each: step, max_fitness, mean_fitness, max_size,
 * mean_size, cum_cell_types, cum_pattern_types, n_active, n_components) are written every
 * sample_interval steps and at the end; the run stops early when mean_size >= stop_mean_size
 * (if > 0) or the grid is empty. init_config (L*L) may be NULL (then n random cells).
 * Returns the number of rows; final_config receives the last configuration.
 */
int schc_run(int L, int n, int k, double mu, double death, int score, int winner_rule, uint64_t seed,
             int steps, int sample_interval, double stop_mean_size,
             const int32_t *init_config, double *rows, int max_rows, int32_t *final_config) {
    world_t *w = world_new(L, k, mu, death, score, winner_rule, seed);
    if (init_config) { for (int i = 0; i < L * L; i++) if (init_config[i]) set_cell(w, i, init_config[i]); }
    else {
        for (int placed = 0; placed < n;) {
            int idx = (int)rng_below(&w->rng, (uint32_t)(L * L));
            if (w->config[idx] == 0) { set_cell(w, idx, 1 + (int32_t)rng_below(&w->rng, (uint32_t)k)); placed++; }
        }
    }
    uint8_t *types_seen = calloc((size_t)k + 1, 1);
    set64_t patterns; set_init(&patterns);
    int nrows = 0;
    for (int t = 0; t <= steps; t++) {
        if (t % sample_interval == 0 || t == steps) {
            double m[6]; metrics(w, m, types_seen, &patterns);
            int ntypes = 0; for (int i = 1; i <= k; i++) ntypes += types_seen[i];
            if (nrows < max_rows) {
                double *r = rows + 9 * (size_t)nrows++;
                r[0] = t; r[1] = m[0]; r[2] = m[1]; r[3] = m[2]; r[4] = m[3];
                r[5] = ntypes; r[6] = (double)patterns.n; r[7] = m[4]; r[8] = m[5];
            }
            if ((stop_mean_size > 0 && m[3] >= stop_mean_size) || w->n_active == 0) break;
        }
        if (t == steps) break;
        onestep(w);
    }
    if (final_config) memcpy(final_config, w->config, sizeof(int32_t) * L * L);
    free(types_seen); free(patterns.keys); world_free(w);
    return nrows;
}

/* testing hooks */
uint32_t schc_component_hash(const int32_t *config, int L, const int32_t *cells, int n, int score, int seed) {
    return component_hash(config, L, cells, n, score, seed);
}
/* one forced contest (given start cells and tie coin) on a copy of config, mu/death as given */
void schc_forced_contest(int32_t *config, int L, int k, double mu, double death, int score, int winner_rule,
                         int start1, int start2, int coin, uint64_t seed) {
    world_t *w = world_new(L, k, mu, death, score, winner_rule, seed);
    for (int i = 0; i < L * L; i++) if (config[i]) set_cell(w, i, config[i]);
    contest(w, 1, start1, start2, coin);
    memcpy(config, w->config, sizeof(int32_t) * L * L);
    world_free(w);
}
