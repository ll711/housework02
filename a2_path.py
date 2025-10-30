"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
#在这个搜索文件中应该包含BFS,DFS,A*等算法
# 在其中这个文件中的所有数据应该先预处理 寻找出变化的大体范围在采用对应的算法
import heapq, time, random
from collections import deque
from typing import List, Tuple, Optional, Dict
Coord = Tuple[int, int]


def path_BFS(start: Coord, end: Coord) -> Optional[List[Coord]]:
    # 这个项目的BFS通过list将每一个图形放到每一个节点上
    #其中list的max,min是标注每个图形的大致范围
    #在这个算法中应该填入每个链表中安全路径的变化即list1,list2,list3中的第一个并以此类推
    rows, cols = len(grid), len(grid[0])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]
    q = deque([start])
    visited = [[False]*cols for _ in range(rows)]
    parent = {}
    visited[start[0]][start[1]] = True
    parent[start] = None

    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] > 0:
                visited[nr][nc] = True
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

def path_DFS(start: Coord, end: Coord) -> Optional[List[Coord]]:
    """
    深度优先搜索（DFS）
    - 若只有 1 个区域：在同一区域内从第一个非零格子到最后一个非零格子；
    - 若 ≥ 2 个区域：从 list1 的任意安全格子出发，寻找能到达 listN 任意安全格子的路径；
    - 使用 8 邻域，与 Get_Graph 一致；返回任意一条可行路径（不保证最短）。
    """
    rows, cols = len(grid), len(grid[0])
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
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                if (nr, nc) not in visited:
                    stack.append(((nr, nc), path + [(nr, nc)]))
    return None

def path_IDDFS(start: Coord, end: Coord) -> Optional[List[Coord]]:
    # 这个项目的BFS通过list将每一个图形放到每一个节点上
    # 其中list的max,min是标注每个图形的大致范围
    #在这个算法中应该采用先进先出+先进后出的思想完成即先广度后深度
    #例如list1,list2,list3中先搜索list1的第一个安全路径变化然后搜索list2的第一个安全路径变化
    #然后搜索list3的第一个安全路径变化然后再回到list1搜索list1的第二个安全路径变化
    # 先提取区域范围
    rows, cols = len(grid), len(grid[0])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def dfs_limited(node: Coord, goal: Coord, limit: int, path: List[Coord]):
        if node == goal:
            return path
        if len(path) > limit:
            return None
        r, c = node
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                if (nr, nc) not in path:
                    res = dfs_limited((nr, nc), goal, limit, path + [(nr, nc)])
                    if res:
                        return res
        return None

    for depth in range(1, rows * cols + 1):
        result = dfs_limited(start, end, depth, [start])
        if result:
            return result
    return None

def path_astar(start: Coord, end: Coord) -> Optional[List[Coord]]:
    # 这个项目的A*通过list将每一个图形放到每一个节点上
    # 其中list的max,min是标注每个图形的大致范围
    # A* 算法是 f(n) = g(n) + h(n)
    # g(n) 是从起点到当前节点的所有安全路径(即路径长度)
    # h(n) 是从当前节点到目标节点的没有完成的路径的估计代价(启发式函数)
    # 在这个算法中应该填入每个链表中安全路径的变化即list1,list2,list3中的第一个并以此类推
    # 收集区域边界
    rows, cols = len(grid), len(grid[0])
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def heuristic(a: Coord, b: Coord) -> float:
        # 曼哈顿距离
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heapq.heappush(open_set, (0, start))
    parent = {start: None}
    g = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == end:
            path = []
            cur = current
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return path[::-1]

        r, c = current
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                new_cost = g[current] + 1
                if (nr, nc) not in g or new_cost < g[(nr, nc)]:
                    g[(nr, nc)] = new_cost
                    f = new_cost + heuristic((nr, nc), end)
                    heapq.heappush(open_set, (f, (nr, nc)))
                    parent[(nr, nc)] = current
    return None
def compare(start: Coord, end: Coord, runs: int = 7, quiet: bool = False):
    # 比较以上四种算法的优缺点
    # 通过计算每个算法的时间长度进行比较
    algos = {
        "BFS": path_BFS,
        "DFS": path_DFS,
        "IDDFS": path_IDDFS,
        "A*": path_astar,
    }

    results: Dict[str, Dict] = {}

    for name, fn in algos.items():
        # 预热一次，减少首次波动
        _ = fn(start, end)

        times: List[float] = []
        last_path: Optional[List[Coord]] = None

        for _ in range(runs):
            t0 = time.perf_counter()
            p = fn(start, end)
            t1 = time.perf_counter()
            times.append(t1 - t0)
            if p is not None:
                last_path = p

        avg = sum(times) / len(times)
        found = last_path is not None
        length = len(last_path) if found else 0

        results[name] = {
            "avg": avg,
            "times": times,
            "found": found,
            "length": length,
            "path": last_path,
        }

    # 找出平均耗时最短者
    fastest = min(results.items(), key=lambda kv: kv[1]["avg"])[0]
    results["fastest"] = fastest

    if not quiet:
        print("\n time costs for each algorithm over ")
        for name in ["BFS", "DFS", "IDDFS", "A*"]:
            r = results[name]
            print(f"{name:<5}: each = " + ", ".join(f"{t:.6f}" for t in r["times"]) +
                  f" | avg = {r['avg']:.6f}s | found={r['found']} | lengh={r['length']}")
        print(f"\n min time cost：{fastest}")

    return results
def min_safe(start,end):
    # 计算从start到end的最短安全路径
    #推测最短应该为A*算法或者IDDFS算法
    candidates = [
        ("A*", path_astar(start, end)),
        ("BFS", path_BFS(start, end)),
        ("IDDFS", path_IDDFS(start, end)),
        ("DFS", path_DFS(start, end)),
    ]

    best_name, best_path = None, None
    for name, p in candidates:
        if p is None:
            continue
        if best_path is None or len(p) < len(best_path):
            best_name, best_path = name, p

    if best_path is not None:
        print(f" Best path by {best_name} ,lengh {len(best_path)}")
    else:
        print(" No path found")
    return best_path

def tester():
    # 测试以上所有函数
    global grid

    grid = [
        [1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 1],
        [1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
    ]

    start = (0, 0)
    end = (9, 9)
    print(f"Start: {start}, End: {end}\n")

    print("BFS:",path_BFS(start,end))
    print("DFS:",path_DFS(start,end))
    print("IDDFS:",path_IDDFS(start,end))
    print("A*:",path_astar(start,end))
    print("Min Safe Path:",min_safe(start,end))
    compare(start, end)


if __name__ == "__main__":
    tester()