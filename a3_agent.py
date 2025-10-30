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
        按注释的优先级对局面打分：
        - 真桥(尤其值为1)强奖励
        - 惩罚 may_hinger 与风险桥位
        - 角 > 边 > 内部
        - 奖励连通块数、奇偶偏置；对总值轻惩罚
        """
        grid = getattr(st, "result", None)
        if not grid or not grid[0]:
            return 0.0

        rows, cols = len(grid), len(grid[0])

        # 若有真桥检测，做一次全量扫描，得到最新真桥集合
        true_bridges_set = set()
        if hasattr(st, "IS_Hinger"):
            try:
                coords = st.IS_Hinger(full_scan=True) or []
                true_bridges_set = set(coords)
            except Exception:
                true_bridges_set = set()

        # 局部工具
        def deg8_local(g, r, c) -> int:
            return self._deg8(g, r, c) if hasattr(self, "_deg8") else sum(
                1
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if not (dr == 0 and dc == 0)
                and 0 <= r + dr < rows
                and 0 <= c + dc < cols
                and g[r + dr][c + dc] > 0
            )

        def count_regions8_local(g) -> int:
            return self._count_regions8(g) if hasattr(self, "_count_regions8") else (
                # 降级实现
                (lambda: (
                    (lambda vis=[]: 0)  # 仅占位，实际不会走到这里
                ))()
            )

        # 统计特征
        total_sum = 0
        edge_cnt = 0
        corner_cnt = 0
        may_hingers = 0
        risky_bridge_like = 0
        true_bridge_ones = 0
        true_bridge_others = 0

        for r in range(rows):
            for c in range(cols):
                v = grid[r][c]
                if v <= 0:
                    continue

                total_sum += v
                is_edge = r == 0 or c == 0 or r == rows - 1 or c == cols - 1
                is_corner = (r in (0, rows - 1)) and (c in (0, cols - 1))
                if is_corner:
                    corner_cnt += 1
                elif is_edge:
                    edge_cnt += 1

                # 真桥奖励（值为1奖励更高）
                if (r, c) in true_bridges_set:
                    if v == 1:
                        true_bridge_ones += 1
                    else:
                        true_bridge_others += 1

                # may_hinger：直线两侧为0且八邻度较高的值为1位置
                if v == 1:
                    lr_zeros = (1 if c - 1 >= 0 and grid[r][c - 1] == 0 else 0) + \
                               (1 if c + 1 < cols and grid[r][c + 1] == 0 else 0)
                    ud_zeros = (1 if r - 1 >= 0 and grid[r - 1][c] == 0 else 0) + \
                               (1 if r + 1 < rows and grid[r + 1][c] == 0 else 0)
                    if (lr_zeros == 2 or ud_zeros == 2) and deg8_local(grid, r, c) >= 2:
                        may_hingers += 1

                    # 风险桥位倾向：一格视野内两侧都有活邻，未来可能成为唯一连接
                    has_left = any(grid[r][cc] > 0 for cc in range(max(0, c - 1), c))
                    has_right = any(grid[r][cc] > 0 for cc in range(c + 1, min(cols, c + 2)))
                    has_up = any(grid[rr][c] > 0 for rr in range(max(0, r - 1), r))
                    has_down = any(grid[rr][c] > 0 for rr in range(r + 1, min(rows, r + 2)))
                    if (has_left and has_right) or (has_up and has_down):
                        risky_bridge_like += 1

        # 连通块与奇偶偏置
        regions = self._count_regions8(grid)

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
                        sz = 0
                        while q:
                            rr, cc = q.popleft()
                            sz += 1
                            for dr, dc in dirs8:
                                nr, nc = rr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc] and grid[nr][nc] > 0:
                                    visited[nr][nc] = True
                                    q.append((nr, nc))
                        sizes.append(sz)
            return sizes

        parity_bias = sum(1 if sz % 2 == 1 else -1 for sz in component_sizes())

        # 线性加权（按照注释的优先级进行）
        score = (
                + 7.0 * true_bridge_ones
                + 4.0 * true_bridge_others
                + 1.5 * regions
                + 0.6 * corner_cnt
                + 0.4 * (edge_cnt)  # 非角边缘
                + 0.3 * parity_bias
                - 3.0 * may_hingers
                - 1.2 * risky_bridge_like
                - 0.04 * total_sum
        )
        return float(score)

    def MiniMax(self, st: state, depth: int = 2, maximizing_player: bool = True) -> Tuple[float, Optional[Coord]]:
        """
        仅扩展两个代表分支：真桥分支 与 非真桥分支。
        若真桥为空，用中性分支与风险分支补齐。
        """
        moves = self._legal_moves(st)
        if depth == 0 or not moves:
            return self.evaluate(st), None

        # 准备真桥集合用于分组
        true_set = set()
        if hasattr(st, "IS_Hinger"):
            try:
                true_set = set(st.IS_Hinger(full_scan=True) or [])
            except Exception:
                true_set = set()

        # 分组
        safe_moves: list[Coord] = []
        neutral_moves: list[Coord] = []
        risky_moves: list[Coord] = []

        rows, cols = len(st.result), len(st.result[0]) if st.result else 0

        def is_may_hinger(r: int, c: int) -> bool:
            v = st.result[r][c]
            if v != 1:
                return False
            lr_zeros = (1 if c - 1 >= 0 and st.result[r][c - 1] == 0 else 0) + \
                       (1 if c + 1 < cols and st.result[r][c + 1] == 0 else 0)
            ud_zeros = (1 if r - 1 >= 0 and st.result[r - 1][c] == 0 else 0) + \
                       (1 if r + 1 < rows and st.result[r + 1][c] == 0 else 0)
            return (lr_zeros == 2 or ud_zeros == 2) and self._deg8(st.result, r, c) >= 2

        for mv in moves:
            if mv in true_set:
                safe_moves.append(mv)
            elif is_may_hinger(*mv):
                risky_moves.append(mv)
            else:
                neutral_moves.append(mv)

        # 代表动作选择与排序（与 _legal_moves 的启发一致）
        def key(mv: Coord):
            r, c = mv
            bridge_bias = 1 if st.result[r][c] == 1 and self._deg8(st.result, r, c) >= 2 else 0
            edge_bias = 1 if r in (0, len(st.result) - 1) or c in (0, len(st.result[0]) - 1) else 0
            return (bridge_bias, edge_bias, -st.result[r][c])

        safe_moves.sort(key=key, reverse=True)
        neutral_moves.sort(key=key, reverse=True)
        risky_moves.sort(key=key, reverse=True)

        # 仅取两类代表
        cand: list[Coord] = []
        if safe_moves:
            cand.append(safe_moves[0])
        if neutral_moves:
            cand.append(neutral_moves[0])
        if not cand and risky_moves:
            cand.append(risky_moves[0])
        if not cand:
            cand.append(moves[0])

        if maximizing_player:
            best_val = float("-inf")
            best_move: Optional[Coord] = None
            for mv in cand:
                child = self._clone_with_move(st, mv)
                val, _ = self.MiniMax(child, depth - 1, maximizing_player=False)
                if val > best_val:
                    best_val, best_move = val, mv
            return best_val, best_move
        else:
            best_val = float("inf")
            best_move: Optional[Coord] = None
            for mv in cand:
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
        先真桥、后中性、最后风险排序；在较深层直接剪掉风险分支，只保留少量中性分支做束搜索。
        再结合标准 alpha-beta 剪枝。
        """
        moves = self._legal_moves(st)
        if depth == 0 or not moves:
            return self.evaluate(st), None

        # 分组准备
        true_set = set()
        if hasattr(st, "IS_Hinger"):
            try:
                true_set = set(st.IS_Hinger(full_scan=True) or [])
            except Exception:
                true_set = set()

        rows, cols = len(st.result), len(st.result[0]) if st.result else 0

        def is_may_hinger(r: int, c: int) -> bool:
            v = st.result[r][c]
            if v != 1:
                return False
            lr_zeros = (1 if c - 1 >= 0 and st.result[r][c - 1] == 0 else 0) + \
                       (1 if c + 1 < cols and st.result[r][c + 1] == 0 else 0)
            ud_zeros = (1 if r - 1 >= 0 and st.result[r - 1][c] == 0 else 0) + \
                       (1 if r + 1 < rows and st.result[r + 1][c] == 0 else 0)
            return (lr_zeros == 2 or ud_zeros == 2) and self._deg8(st.result, r, c) >= 2

        safe_moves: list[Coord] = []
        neutral_moves: list[Coord] = []
        risky_moves: list[Coord] = []

        for mv in moves:
            if mv in true_set:
                safe_moves.append(mv)
            elif is_may_hinger(*mv):
                risky_moves.append(mv)
            else:
                neutral_moves.append(mv)

        def key(mv: Coord):
            r, c = mv
            bridge_bias = 1 if st.result[r][c] == 1 and self._deg8(st.result, r, c) >= 2 else 0
            edge_bias = 1 if r in (0, len(st.result) - 1) or c in (0, len(st.result[0]) - 1) else 0
            return (bridge_bias, edge_bias, -st.result[r][c])

        safe_moves.sort(key=key, reverse=True)
        neutral_moves.sort(key=key, reverse=True)
        risky_moves.sort(key=key, reverse=True)

        # 深度越深，越果断剪掉风险，并限制中性束宽
        beam_neutral = 2 if depth >= 2 else 4
        ordered: list[Coord] = []
        ordered.extend(safe_moves)
        ordered.extend(neutral_moves[:beam_neutral])
        if depth <= 1:  # 只在浅层允许看一点风险分支
            ordered.extend(risky_moves[:1])

        if maximizing_player:
            best_val = float("-inf")
            best_move: Optional[Coord] = None
            for mv in ordered:
                child = self._clone_with_move(st, mv)
                val, _ = self.AlphaBeta(child, depth - 1, alpha, beta, maximizing_player=False)
                if val > best_val:
                    best_val, best_move = val, mv
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val, best_move
        else:
            best_val = float("inf")
            best_move: Optional[Coord] = None
            for mv in ordered:
                child = self._clone_with_move(st, mv)
                val, _ = self.AlphaBeta(child, depth - 1, alpha, beta, maximizing_player=True)
                if val < best_val:
                    best_val, best_move = val, mv
                beta = min(beta, best_val)
                if beta <= alpha:
                    break
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