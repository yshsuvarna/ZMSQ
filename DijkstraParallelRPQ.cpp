// CS309
#include "RelaxedPriorityQueue.h"
#include <bits/stdc++.h>
#include <thread>
using namespace std;

static const int NUM_THREADS = 8;
const int INF = 1e9;

int n, m;
vector<vector<pair<int, int>>> adj;

RelaxedPriorityQueue<pair<int, int>> q;
vector<int> d;

void relax(int i, int u, int npt) {
    int sz = adj[u].size();
    int st = i*npt;
    int en = -1;
    if(i == NUM_THREADS - 1) {
        en = sz;
    } else {
        en = st + npt;
    }
    for(int v = st; v < en; v++) {
        int ver = adj[u][v].first;
        int len = adj[u][v].second;
        if(d[u] + len < d[ver]) {
            d[ver] = d[u] + len;
            q.push({-d[ver], ver});
        }
    }
}

thread tt[NUM_THREADS];
void dijkstra(int s) {
    d.assign(n, INF);
    d[s] = 0;
    q.push({0, s});
    while (!q.empty()) {
        pair<int, int> u;
        q.pop(u);
        int v = u.second;
        int d_v = -u.first;
        if (d_v != d[v])
            continue;
        int sz = adj[v].size();
        if(sz >= NUM_THREADS) {
            int v_per_thread = sz/(NUM_THREADS);
            for(int i = 0; i < NUM_THREADS; i++) {
                tt[i] = thread(relax, i, v, v_per_thread);
            }
            for(int i = 0; i < NUM_THREADS; i++) {
                tt[i].join();
            }
        } else {
            for(auto e: adj[v]) {
                int to = e.first;
                int len = e.second;
                if(d[v] + len < d[to]) {
                    d[to] = d[v] + len;
                    q.push({-d[to], to});
                }
            }
        }
    }
}

int main() {
    cin >> n >> m;
    adj = vector<vector<pair<int, int>>>(n);
    for(int i = 0; i < m; i++) {
        int a, b, D;
        cin >> a >> b >> D;
        a--; b--;
        adj[a].push_back({b, D});
        adj[b].push_back({a, D});
    }
    auto start = std::chrono::high_resolution_clock::now();
    dijkstra(0);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    // Time taken for the process
    int time_taken = duration.count();
    cout << time_taken << endl;
}
