# Parallel Dijkstra with Relaxed Priority Queue

This repository contains an experimental C++ implementation of Dijkstra's shortest path
algorithm that leverages a custom **Relaxed Priority Queue (RPQ)** to parallelise work.
The RPQ—referred to in the code as **ZMSQ**—is a concurrent priority queue that trades
strict ordering for reduced synchronisation overhead, allowing multiple threads to operate
on the queue at once.

## ZMSQ overview
`ZMSQ` is implemented in `RelaxedPriorityQueue.h` and organises items in a multi‑level
"mound" where each node holds a small unordered list. Push operations walk down the tree
until a slot with a larger key is found, while pop operations remove from the root list
and rebalance when necessary. The data structure relies on light‑weight spin locks and
atomic pointers, plus an optional shared buffer for batched pops, so most operations touch
disjoint parts of the mound.

Because it only approximates global ordering, ZMSQ lets many producers and consumers make
progress concurrently, avoiding the single lock or strict heap property of a classical
priority queue. For algorithms that can tolerate minor out‑of‑order processing—such as
Dijkstra’s algorithm used here—this relaxation can offer better throughput on multi‑core
machines.

Compared with a traditional binary heap protected by a single mutex, ZMSQ reduces
contention and scales better with thread count by allowing operations to proceed on
different parts of the structure simultaneously.

## Contents
- `DijkstraParallelRPQ.cpp` – demonstration program running Dijkstra's algorithm.
  It divides edge relaxations across `NUM_THREADS` worker threads and stores work in the RPQ.
- `RelaxedPriorityQueue.h` – header-only implementation of the relaxed priority queue
  built on a hierarchical "mound" structure using atomics and spin locks.
- `SpinLoc.h` – simple spin lock used by the queue for fine‑grained synchronisation.
- `Gr_06 CS 309 Documentation.pdf` – original project documentation.

## Building
The program uses C++17 features and the POSIX threads library:

```bash
g++ -std=c++17 DijkstraParallelRPQ.cpp -pthread -O2 -o dijkstra
```

## Running
The compiled executable expects a graph in the following format on standard input:

```
<n> <m>
<u1> <v1> <w1>
...
<um> <vm> <wm>
```

Vertices are numbered from 1.  The program runs Dijkstra starting from vertex 1 and
prints the time taken (in microseconds) to standard output.

Example:

```bash
printf "3 3\n1 2 1\n2 3 1\n1 3 4\n" | ./dijkstra
```

## Notes
- `NUM_THREADS` in `DijkstraParallelRPQ.cpp` controls the level of parallelism
  for edge relaxation.
- The relaxed priority queue may return elements out of strict priority order,
  but algorithms such as Dijkstra's can tolerate this relaxation to gain speed
  on multi‑core systems.
