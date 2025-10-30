"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
# genet ic algorithm
import collections

from typing import List, Tuple, Optional
from MyList import MyList as mylist
from a1_state import State as state

Coord = Tuple[int, int]

class Agent:
    def __init__(self):
        self.name = "B20"
        self.size = "m,n"
        self.model = ["minimax", "alphabeta"]

    def __str__(self):
        return self.name + self.size

    def move(self, state: state, mode: str = "alphabeta") -> Optional[Coord]:
        grid = getattr(state, "result", None)
        if not grid:
            return None
        fallback_moves = [(r, c)
                          for r, row in enumerate(grid)
                          for c, v in enumerate(row) if v > 0]
        if not fallback_moves:
            return None

        m = (self.model or "alphabeta").lower()
        try:
            if m in ("alphabeta", "alpha-beta", "alpha", "ab"):
                _, mv = self.AlphaBeta(
                    state, depth=3, alpha=float("-inf"), beta=float("inf"), maximizing_player=True
                )
            elif m in ("minimax", "mini", "mm"):
                _, mv = self.MiniMax(state, depth=3, maximizing_player=True)
            else:
                # 未知策略名时默认使用 alphabeta
                _, mv = self.AlphaBeta(
                    state, depth=3, alpha=float("-inf"), beta=float("inf"), maximizing_player=True
                )
            return mv if mv is not None else fallback_moves[0]
        except Exception:
            # 搜索异常时回退到首个合法坐标
            return fallback_moves[0]

    def evaluate(self, st: state) -> float:
        """
        局面评估：
        - 鼓励“桥倾向”(bridge_like)与区域数(regions)，鼓励边缘占位(edge_pos)
        - 惩罚潜在桥(may_hingers)与总权重(total_sum)
        - 对连通块大小的奇偶做轻微偏置(parity_bias)
        """
        grid = getattr(st, "result", None)
        if not grid or not grid[0]:
            return 0.0

        rows, cols = len(grid), len(grid[0])
        total_sum = 0  # 所有正值之和（惩罚）
        edge_pos = 0  # 边缘的正值格数量（奖励）
        may_hingers = 0  # 潜在桥位置计数（惩罚）
        bridge_like = 0  # 桥倾向位置计数（奖励）

        # 若类中实现了 _deg8 与 _count_regions8，则复用；否则在本函数内降级实现
        def _deg8_local(g, r, c) -> int:
            deg = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and g[nr][nc] > 0:
                        deg += 1
            return deg

        def _regions8_local(g) -> int:
            from collections import deque
            vis = [[False] * cols for _ in range(rows)]
            dirs8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
            cnt = 0
            for i in range(rows):
                for j in range(cols):
                    if g[i][j] > 0 and not vis[i][j]:
                        cnt += 1
                        q = deque([(i, j)])
                        vis[i][j] = True
                        while q:
                            r, c = q.popleft()
                            for dr, dc in dirs8:
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < rows and 0 <= nc < cols and g[nr][nc] > 0 and not vis[nr][nc]:
                                    vis[nr][nc] = True
                                    q.append((nr, nc))
            return cnt

        deg8 = getattr(self, "_deg8", None)
        if not callable(deg8):
            deg8 = _deg8_local
        count_regions8 = getattr(self, "_count_regions8", None)
        if not callable(count_regions8):
            count_regions8 = _regions8_local

        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v <= 0:
                    continue

                total_sum += v
                if r == 0 or c == 0 or r == rows - 1 or c == cols - 1:
                    edge_pos += 1

                if v == 1:
                    # 潜在桥（左右为0或上下为0；且八邻域度>=2）
                    lr_zeros = (1 if c - 1 >= 0 and grid[r][c - 1] == 0 else 0) + \
                               (1 if c + 1 < cols and grid[r][c + 1] == 0 else 0)
                    ud_zeros = (1 if r - 1 >= 0 and grid[r - 1][c] == 0 else 0) + \
                               (1 if r + 1 < rows and grid[r + 1][c] == 0 else 0)
                    if (lr_zeros == 2 or ud_zeros == 2) and deg8(grid, r, c) >= 2:
                        may_hingers += 1

                    # 桥倾向：一格视野内在该方向两侧都有活邻
                    has_left = any(grid[r][cc] > 0 for cc in range(max(0, c - 1), c))
                    has_right = any(grid[r][cc] > 0 for cc in range(c + 1, min(cols, c + 2)))
                    has_up = any(grid[rr][c] > 0 for rr in range(max(0, r - 1), r))
                    has_down = any(grid[rr][c] > 0 for rr in range(r + 1, min(rows, r + 2)))
                    if (has_left and has_right) or (has_up and has_down):
                        bridge_like += 1

        regions = count_regions8(grid)

        # 连通块大小奇偶性偏置（奇数+1，偶数-1）
        def component_sizes() -> list[int]:
            from collections import deque
            visited = [[False] * cols for _ in range(rows)]
            dirs8 = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
            sizes: list[int] = []
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] > 0 and not visited[i][j]:
                        q = deque([(i, j)])
                        visited[i][j] = True
                        size = 0
                        while q:
                            rr, cc = q.popleft()
                            size += 1
                            for dr, dc in dirs8:
                                nr, nc = rr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] > 0:
                                    visited[nr][nc] = True
                                    q.append((nr, nc))
                        sizes.append(size)
            return sizes

        parity_bias = 0
        for sz in component_sizes():
            parity_bias += (1 if sz % 2 == 1 else -1)

        # 线性加权评分（可按需要微调权重）
        score = (
                + 5.0 * bridge_like
                - 3.0 * may_hingers
                + 1.0 * regions
                + 0.5 * edge_pos
                - 0.05 * total_sum
                + 0.3 * parity_bias
        )
        return float(score)

    def MiniMax(self, st: state, depth: int = 2, maximizing_player: bool = True) -> Tuple[float, Optional[Coord]]:
        """
        经典极大极小：
        - 终止：深度==0 或无合法着法
        - 递归：遍历合法着法，分别克隆落子后递归评估
        """
        moves = self._legal_moves(st)
        if depth == 0 or not moves:
            return self.evaluate(st), None

        if maximizing_player:
            best_val = float("-inf")
            best_move: Optional[Coord] = None
            for mv in moves:
                child = self._clone_with_move(st, mv)
                val, _ = self.MiniMax(child, depth - 1, maximizing_player=False)
                if val > best_val:
                    best_val, best_move = val, mv
            return best_val, best_move
        else:
            best_val = float("inf")
            best_move: Optional[Coord] = None
            for mv in moves:
                child = self._clone_with_move(st, mv)
                val, _ = self.MiniMax(child, depth - 1, maximizing_player=True)
                if val < best_val:
                    best_val, best_move = val, mv
            return best_val, best_move

    def AlphaBeta(self,
                  st: state,
                  depth: int,
                  alpha: float,
                  beta: float,
                  maximizing_player: bool = True) -> Tuple[float, Optional[Coord]]:
        """
        Alpha-Beta 剪枝：
        - 与 MiniMax 相同终止条件
        - 通过 alpha/beta 界提前剪去不可能改善的分支
        """
        moves = self._legal_moves(st)
        if depth == 0 or not moves:
            return self.evaluate(st), None

        if maximizing_player:
            best_val = float("-inf")
            best_move: Optional[Coord] = None
            for mv in moves:
                child = self._clone_with_move(st, mv)
                val, _ = self.AlphaBeta(child, depth - 1, alpha, beta, maximizing_player=False)
                if val > best_val:
                    best_val, best_move = val, mv
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break  # 剪枝
            return best_val, best_move
        else:
            best_val = float("inf")
            best_move: Optional[Coord] = None
            for mv in moves:
                child = self._clone_with_move(st, mv)
                val, _ = self.AlphaBeta(child, depth - 1, alpha, beta, maximizing_player=True)
                if val < best_val:
                    best_val, best_move = val, mv
                beta = min(beta, best_val)
                if beta <= alpha:
                    break  # 剪枝
            return best_val, best_move
    # ========= 辅助：生成落子、克隆走子、连通性与邻域 =========

    def _legal_moves(self, st: state) -> List[Coord]:
        g = st.result
        moves: List[Coord] = []
        for r in range(len(g)):
            for c in range(len(g[0]) if g else 0):
                if g[r][c] > 0:
                    moves.append((r, c))

        # 简单启发排序：优先考虑桥倾向、再考虑边界
        def key(mv: Coord):
            r, c = mv
            bridge_bias = 1 if g[r][c] == 1 and self._deg8(g, r, c) >= 2 else 0
            edge_bias = 1 if r in (0, len(g) - 1) or c in (0, len(g[0]) - 1) else 0
            return (bridge_bias, edge_bias, -g[r][c])

        moves.sort(key=key, reverse=True)
        return moves

    def _clone_with_move(self, st: state, mv: Coord) -> state:
        r, c = mv
        new_grid = [row[:] for row in st.result]
        if new_grid[r][c] > 0:
            new_grid[r][c] -= 1
        child = state(new_grid)
        child.Get_Graph()
        return child

    def _count_regions8(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        rows, cols = len(grid), len(grid[0])
        vis = [[False] * cols for _ in range(rows)]
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        def bfs(sr: int, sc: int):
            from collections import deque
            q = deque([(sr, sc)])
            vis[sr][sc] = True
            while q:
                r, c = q.popleft()
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not vis[nr][nc] and grid[nr][nc] > 0:
                        vis[nr][nc] = True
                        q.append((nr, nc))

        cnt = 0
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] > 0 and not vis[r][c]:
                    cnt += 1
                    bfs(r, c)
        return cnt

    def _deg8(self, grid: List[List[int]], r: int, c: int) -> int:
        rows, cols = len(grid), len(grid[0]) if grid else 0
        deg = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] > 0:
                    deg += 1
        return deg
    def May_Hinger(self, node=None, full_scan=False, return_new_bridge=False):
        """
        优化后的May_Hinger方法：使用坐标列表比较替代数量比较。准确检测潜在桥梁的位置变化
        修改后的May_Hinger方法，返回字符串"true"/"false"表示是否产生新可能桥梁
        检查潜在桥梁状态，使用局部坐标遍历，仅在需要时转换为全局坐标
        :param node: 可选，指定要检查的节点（用于增量更新）
        :param full_scan: 是否进行全量扫描（首次调用时设置为True）
        :param return_new_bridge: 是否返回新桥标志
        :return: 如果return_new_bridge为True，返回"true"或"false"
        """
        if full_scan:
            # 全量扫描逻辑保持不变
            current_node = self.state.mylist.head
            while current_node is not None:
                self._process_node_may_hinger(current_node)
                current_node = current_node.next
            return None

        elif node is not None:
            # 保存旧的潜在桥梁坐标列表（使用局部坐标）
            old_array_data = node.get_array_data()
            old_bridge_coords = set()   # 使用集合而非列表提高性能

            if old_array_data:
                for i_local in range(len(old_array_data)):
                    for j_local in range(len(old_array_data[0])):
                        if old_array_data[i_local][j_local] == 1:
                            old_bridge_coords.add((i_local, j_local))

            # 处理节点，更新潜在桥梁标记
            self._process_node_may_hinger(node)

            # 获取新的潜在桥梁坐标列表
            new_array_data = node.get_array_data()
            new_bridge_coords = set()   # 使用集合而非列表提高性能

            if new_array_data:
                for i_local in range(len(new_array_data)):
                    for j_local in range(len(new_array_data[0])):
                        if new_array_data[i_local][j_local] == 1:
                            new_bridge_coords.add((i_local, j_local))

            # 使用集合操作检测新桥梁：检查新坐标集合中有但旧坐标集合中没有的坐标
            new_bridges = new_bridge_coords - old_bridge_coords
            has_new_bridge = len(new_bridges) > 0

            """
            # 可选：记录新桥梁的详细信息（用于调试）
            if has_new_bridge and return_new_bridge:
                print(f"检测到新潜在桥梁: {new_bridges}")
            """

            if return_new_bridge:
                return "true" if has_new_bridge else "false"

        return None

    def _process_node_may_hinger(self, node):
        """
        处理单个节点的潜在桥梁检测，使用局部坐标
        """
        # 获取节点的网格数据（局部坐标）
        node_grid = node.get_grid().data

        # 获取节点网格尺寸
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # 获取或初始化array_data
        array_data = node.get_array_data()
        if not array_data:
            array_data = [[0] * node_cols for _ in range(node_rows)]

        # 清空之前的潜在桥梁标记
        for i in range(node_rows):
            for j in range(node_cols):
                array_data[i][j] = 0

            # 标记潜在桥梁（只检查计数器值为1的单元格）
            for i_local in range(node_rows):
                for j_local in range(node_cols):
                    if node_grid[i_local][j_local] == 1:
                        if self._check_potential_hinger_local(node_grid, i_local, j_local, node_rows, node_cols):
                            array_data[i_local][j_local] = 1

            # 更新节点的array_data
            node.set_grid_data(array_data)

        def _check_potential_hinger_local(self, grid, i, j, rows, cols):
            """
            在局部坐标中检查单个位置是否为潜在桥梁
            """
            # 检查直接相邻的左右和上下方向
            directions = [
                [(0, -1), (0, 1)],  # 左右方向（同一行）
                [(-1, 0), (1, 0)]  # 上下方向（同一列）
            ]

            for dir_pair in directions:
                zero_count = 0
                for dr, dc in dir_pair:
                    r_adj = i + dr
                    c_adj = j + dc
                    if 0 <= r_adj < rows and 0 <= c_adj < cols:
                        if grid[r_adj][c_adj] == 0:
                            zero_count += 1
                if zero_count >= 2:
                    return True

            return False

    def decision_tree(self, event_x: int, event_y: int, grid_row: int, grid_col: int):
        """
        决策树方法：处理鼠标点击事件，协调调用其他函数
        :param event_x: 鼠标点击的像素坐标x
        :param event_y: 鼠标点击的像素坐标y
        :param grid_row: 点击的网格行坐标
        :param grid_col: 点击的网格列坐标
        :return: 无返回值，但会更新内部状态
        """
        print("开始决策树处理流程")

        # 获取点击位置计数器值
        current_value = self.state.grid[grid_row][grid_col]
        print(f"坐标({grid_row}, {grid_col})的计数器值: {current_value}")

        # 决策点1: 计数器为0或>2?
        if current_value == 0 or current_value > 2:
            print("决策点1: 计数器为0或大于2，结束处理")
            return

        # 决策点2: 计数器为1或2，调用change_data修改数据
        print("决策点2: 计数器为1或2，调用change_data修改数据")
        success = self.state.Change_Data(grid_row, grid_col)
        if not success:
            print("数据修改失败，结束处理")
            return

        # 获取受影响节点
        affected_node = self.state.Search_Node(grid_row, grid_col)
        if affected_node is None:
            print("警告: 未找到受影响节点，结束处理")
            return

        print(
            f"找到受影响节点: 范围({affected_node.get_min_x()},{affected_node.get_min_y()})到({affected_node.get_max_x()},{affected_node.get_max_y()})")

        # 决策点3: 调用May_Hinger → 返回new_bridge标志
        print("决策点3: 调用May_Hinger检查潜在桥梁")
        new_bridge = self.May_Hinger(affected_node, return_new_bridge=True)
        print(f"May_Hinger返回new_bridge = {new_bridge}")

        # 决策点4: new_bridge为"true"?
        if new_bridge != "true":
            print("决策点4: new_bridge不为'true'，结束处理")
            return

        print("决策点4: new_bridge为'true'，继续处理")

        # 决策点5: 调用IS_Hinger → 更新全局真桥列表
        print("决策点5: 调用IS_Hinger判断真桥")
        self.state.IS_Hinger(node=affected_node)

        # 更新桥梁数量
        self.state.numHingers()
        print(f"决策树处理完成，当前桥梁数量: {self.state.hinger_count}")

    def MCTS(self):
        pass
def teser():
    """
    在几组小网格上测试 MiniMax 与 AlphaBeta 的返回分数与推荐落子。
    注意：需要先完成 Agent.evaluate/MiniMax/AlphaBeta 内的占位符实现。
    """
    def show_grid(g):
        for r in g:
            print(" ".join(f"{v:2d}" for v in r))
        print()

    cases = [
        # 案例1：一条带转折的连通带
        (
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
            ],
            2, 3
        ),
        # 案例2：两块区域+部分2值，便于产生/消除桥
        (
            [
                [2, 1, 0, 0, 1],
                [0, 1, 1, 0, 1],
                [0, 0, 2, 0, 1],
                [0, 0, 1, 1, 1],
                [0, 0, 0, 0, 1],
            ],
            2, 3
        ),
        # 案例3：更稀疏，利于观察边界优先与奇偶影响
        (
            [
                [1, 0, 1, 0],
                [0, 2, 0, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            2, 3
        ),
    ]
    more_cases = [
        # 案例4：十字交叉（中心格往往是桥）
        (
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            2, 3
        ),
        # 案例5：水平走廊（1x6）
        (
            [
                [1, 1, 1, 1, 1, 1],
            ],
            2, 3
        ),
        # 案例6：垂直走廊（6x1）
        (
            [
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
            ],
            2, 3
        ),
        # 案例7：环形包围，中间留孔（桥较少，边界显著）
        (
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1],
            ],
            3, 4
        ),
        # 案例8：棋盘格（8 邻域下连通性强）
        (
            [
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
            ],
            2, 3
        ),
        # 案例9：两个相距较远的区域（分离岛）
        (
            [
                [1, 1, 0, 0, 0, 1, 1],
                [1, 1, 0, 0, 0, 1, 1],
            ],
            2, 3
        ),
        # 案例10：含较多的 2，测试减子对桥/安全路径的影响
        (
            [
                [2, 0, 2, 0, 2],
                [0, 2, 0, 2, 0],
                [2, 0, 2, 0, 2],
            ],
            2, 3
        ),
        # 案例11：显式构造 may_hinger（中心左右为 0，斜对角有支撑）
        (
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            2, 3
        ),
        # 案例12：T 字分叉（分支选择与桥评估）
        (
            [
                [0, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
            ],
            2, 3
        ),
    ]

    cases += more_cases
    from a1_state import State as state
    ag = Agent()

    for idx, (grid, d_mm, d_ab) in enumerate(cases, 1):
        print(f"=== Case {idx} ===")
        print("初始网格：")
        show_grid(grid)

        st = state(grid)
        st.Get_Graph()

        # MiniMax
        try:
            mm_score, mm_move = ag.MiniMax(st, depth=d_mm, maximizing_player=True)
            print(f"MiniMax(depth={d_mm}) -> score={mm_score:.3f}, move={mm_move}")
        except Exception as e:
            print(f"MiniMax 执行异常: {e}")
            mm_move = None

        # Alpha-Beta
        try:
            ab_score, ab_move = ag.AlphaBeta(st, depth=d_ab, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
            print(f"AlphaBeta(depth={d_ab}) -> score={ab_score:.3f}, move={ab_move}")
        except Exception as e:
            print(f"AlphaBeta 执行异常: {e}")
            ab_move = None

        # 应用各自推荐一步并打印效果
        def apply_and_show(move, tag):
            if move is None:
                return
            try:
                child = ag._clone_with_move(st, move)
                print(f"{tag} 应用推荐落子 {move} 后网格：")
                show_grid(child.result)
            except Exception as e:
                print(f"{tag} 应用落子异常: {e}")

        apply_and_show(mm_move, "MiniMax")
        apply_and_show(ab_move, "AlphaBeta")

    print("测试完成。")


if __name__ == "__main__":
    # 直接运行本文件时执行测试
    teser()