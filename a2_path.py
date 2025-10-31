"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
# This search file should include algorithms such as BFS, DFS, A*, etc.
# For all the data in this file, it should be preprocessed first to identify the general range of changes and then apply the corresponding algorithm.
import heapq
from collections import deque
from a1_state import State

Coord = tuple[int, int]


def _is_safe(state: State, r: int, c: int) -> bool:
    """
    Check whether cell (r, c) is safe to move to.

    A cell is considered safe if:
      - It is within bounds and active (value > 0),
      - And clicking it does NOT create a hinger (when we can check that).
    If hinger information is unavailable (e.g. Search_Node() returns None),
    we fall back to simple 0/1 logic (1 = safe, 0 = unsafe).
    """
    if not (0 <= r < state.m and 0 <= c < state.n):
        return False
    if state.result[r][c] <= 0:
        return False

    # Try to find which node owns this coordinate
    node = None
    try:
        node = state.Search_Node(r, c)
    except Exception:
        node = None

    # Fallback: if we can't find any node, treat as safe cell (normal 1/0 behavior)
    if node is None:
        return True

    # Convert global -> local coordinates
    min_x, min_y = node.get_min_x(), node.get_min_y()
    i_local = r - (min_y - 1)
    j_local = c - (min_x - 1)

    array_data = node.get_array_data()
    rows = len(array_data)
    cols = len(array_data[0]) if rows > 0 else 0

    # Out of local grid range? Safe fallback.
    if not (0 <= i_local < rows and 0 <= j_local < cols):
        return True

    # Not a potential bridge → safe
    if array_data[i_local][j_local] != 1:
        return True

    # Potential bridge → simulate click
    try:
        creates_hinger = state._check_hinger_creates_new_region_local(node, i_local, j_local)
    except Exception:
        # If the check function fails, be conservative
        creates_hinger = True

    return not creates_hinger

def _cell_risk(state: State, r: int, c: int) -> int:
    """
    Risk of stepping on (r, c): number of 1-cells in its 8-neighborhood.
    Assumes state.result[r][c] > 0 and (r,c) is safe by _is_safe.
    """
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    risk = 0
    for dr, dc in dirs:
        nr, nc = r + dr, c + dc
        if 0 <= nr < state.m and 0 <= nc < state.n and state.result[nr][nc] > 0:
            risk += 1
    return risk


def _path_risk(state: State, path: list[Coord]) -> int:
    """
    Total risk of a path = sum of _cell_risk for each cell on the path.
    (If you want to disregard the starting point, replace "range" with "path[1:]")
    """
    if not path:
        return 0
    return sum(_cell_risk(state, r, c) for r, c in path)

# ==============================================================
# BFS Safe Path
# ==============================================================

def path_BFS(start: Coord, end: Coord, state: State):
    """Breadth-First Search (BFS) safe path algorithm."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    q = deque([start])
    parent = {start: None}
    visited = set([start])

    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and _is_safe(state, nr, nc):
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

    if end not in parent:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


# ==============================================================
# DFS Safe Path
# ==============================================================

def path_DFS(start: Coord, end: Coord, state: State):
    """Depth-First Search (DFS) safe path algorithm."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    stack = [(start, [start])]
    visited = set()

    while stack:
        (r, c), path = stack.pop()
        if (r, c) == end:
            return path
        if (r, c) in visited:
            continue
        visited.add((r, c))

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and _is_safe(state, nr, nc):
                stack.append(((nr, nc), path + [(nr, nc)]))
    return None


# ==============================================================
# IDDFS Safe Path
# ==============================================================

def path_IDDFS(start: Coord, end: Coord, state: State, max_depth=50):
    """Iterative Deepening DFS (IDDFS) safe path algorithm."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def dls(node, end, depth, path, visited):
        if node == end:
            return path
        if depth <= 0:
            return None

        r, c = node
        visited.add(node)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in visited and _is_safe(state, nr, nc):
                new_path = dls((nr, nc), end, depth - 1, path + [(nr, nc)], visited)
                if new_path:
                    return new_path
        return None

    for limit in range(1, max_depth + 1):
        visited = set()
        result = dls(start, end, limit, [start], visited)
        if result:
            return result
    return None


# ==============================================================
# A* Safe Path
# ==============================================================

def path_astar(start: Coord, end: Coord, state: State):
    """A* safe path algorithm using Manhattan heuristic."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def h(a: Coord, b: Coord):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    parent = {start: None}
    g = {start: 0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == end:
            break

        r, c = cur
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if not _is_safe(state, nr, nc):
                continue

            new_cost = g[cur] + 1
            if (nr, nc) not in g or new_cost < g[(nr, nc)]:
                g[(nr, nc)] = new_cost
                f = new_cost + h((nr, nc), end)
                parent[(nr, nc)] = cur
                heapq.heappush(open_set, (f, (nr, nc)))

    if end not in parent:
        return None

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


# ==============================================================
# Compare Algorithms and Select Shortest Safe Path
# ==============================================================

def min_safe(start: Coord, end: Coord, state: State,
             return_algo: bool = False, criterion: str = "risk"):
    """
    Choose the best safe path across the four algorithms.

    criterion:
        "risk"   -> choose the path with the lowest total risk (default)
        "length" -> choose the shortest path (fallback to old behavior)

    If return_algo=True, returns (best_path, algo_func, score),
    where score is risk (if criterion="risk") or length (if "length").
    """
    # Run all four
    results = [
        (path_BFS,   path_BFS(start, end, state)),
        (path_DFS,   path_DFS(start, end, state)),
        (path_IDDFS, path_IDDFS(start, end, state)),
        (path_astar, path_astar(start, end, state)),
    ]

    best_func = None
    best_path = None
    best_score = None

    for func, p in results:
        if not p:
            continue
        score = _path_risk(state, p) if criterion == "risk" else len(p)
        if best_score is None or score < best_score or (score == best_score and len(p) < len(best_path)):
            best_func = func
            best_path = p
            best_score = score

    if return_algo:
        return best_path, best_func, best_score
    return best_path


# ==============================================================
# Tester
# ==============================================================

if __name__ == "__main__":
    print("=== Safe Path Tester ===")

    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 0, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 0, 1, 1],
    ]

    state = State(grid)
    if hasattr(state, "Get_Graph"):
        state.Get_Graph()

    start, end = (0, 0), (4, 4)

    # Run all algorithms
    bfs_path = path_BFS(start, end, state)
    dfs_path = path_DFS(start, end, state)
    iddfs_path = path_IDDFS(start, end, state)
    astar_path = path_astar(start, end, state)

    # Compute risk for each path
    bfs_risk = _path_risk(state, bfs_path)
    dfs_risk = _path_risk(state, dfs_path)
    iddfs_risk = _path_risk(state, iddfs_path)
    astar_risk = _path_risk(state, astar_path)

    # Choose best path based on risk
    best_path, best_func, best_risk = min_safe(start, end, state, return_algo=True, criterion="risk")

    # Print all results with risk values
    print(f"BFS Path:   {bfs_path}\n  → Total Risk = {bfs_risk}")
    print(f"DFS Path:   {dfs_path}\n  → Total Risk = {dfs_risk}")
    print(f"IDDFS Path: {iddfs_path}\n  → Total Risk = {iddfs_risk}")
    print(f"A* Path:    {astar_path}\n  → Total Risk = {astar_risk}")

    algo_name = best_func.__name__ if best_func else "None"
    print(f"\nBest (lowest-risk) path by {algo_name}:")
    print(f"  Path = {best_path}")
    print(f"  Total Risk = {best_risk}")