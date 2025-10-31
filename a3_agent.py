"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
# genet ic algorithm
import collections

from typing import Optional, List, Tuple, Optional, Set

import a1_state
from MyList import MyList as mylist
from a1_state import State as state

# 如果 ListNode 在另一个模块中，需要导入
try:
    from MyList import ListNode
except ImportError:
    # 如果 MyList 在同一文件中定义，使用全局定义
    pass

Coord = Tuple[int, int]

class Agent:
    def __init__(self, state=None):
        self.name = "B20"
        self.size = "m,n"
        self.model = ["minimax", "alphabeta"]
        self.state = state #储存State实例引用

    def __str__(self):
        return self.name + self.size
    def move(self, st: state, mode: str = "alphabeta") -> Optional[Coord]:
        if not hasattr(st, "result") or not st.result:
            return None
        actives = [(r, c) for r, row in enumerate(st.result) for c, v in enumerate(row) if v > 0]
        if not actives:
            return None
        m = (mode or "alphabeta").lower()
        if m in ("minimax", "mm", "mini"):
            _, mv = self.MiniMax(st, depth=1, maximizing_player=True)
        else:
            _, mv = self.AlphaBeta(st, depth=2, alpha=float("-inf"), beta=float("inf"), maximizing_player=True)
        return mv if mv is not None else actives[0]

    # ===== 评估：多真桥、多>2、少可能桥、总和小 =====
    def evaluate(self, st: state) -> float:
        self._ensure_graph(st)
        # 刷新可能桥与真桥
        self.May_Hinger(st, full_scan=True)
        try:
            st.IS_Hinger(full_scan=True)
            true_h = st.Get_hinger_global_coords() or []
        except Exception:
            true_h = []
        poss = self._list_possible_bridge_globals_on(st)
        actives = [(r, c) for r, row in enumerate(st.result) for c, v in enumerate(row) if v > 0]
        num_gt2 = sum(1 for (r, c) in actives if st.result[r][c] > 2)
        total_sum = sum(st.result[r][c] for (r, c) in actives)
        score = (
            6.0 * len(true_h) +
            1.5 * num_gt2 -
            1.0 * len(poss) -
            0.01 * total_sum
        )
        return float(score)

    # ===== MiniMax：按题述优先级直接取首选 =====
    def MiniMax(self, st: state, depth: int = 1, maximizing_player: bool = True):
        cands = self._ordered_candidates(st)
        best = cands[0] if cands else None
        return self.evaluate(st), best

    # ===== Alpha-Beta：相同候选顺序展开与剪枝 =====
    def AlphaBeta(self, st: state, depth: int = 2,
                  alpha: float = float("-inf"), beta: float = float("inf"),
                  maximizing_player: bool = True):
        if depth <= 0:
            return self.evaluate(st), None
        cands = self._ordered_candidates(st)
        if not cands:
            return self.evaluate(st), None

        best_move = None
        if maximizing_player:
            value = float("-inf")
            for mv in cands:
                child = self._clone_with_move(st, mv)
                score, _ = self.AlphaBeta(child, depth - 1, alpha, beta, False) if depth > 1 else (self.evaluate(child), None)
                if score > value:
                    value, best_move = score, mv
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value, best_move
        else:
            value = float("inf")
            for mv in cands:
                child = self._clone_with_move(st, mv)
                score, _ = self.AlphaBeta(child, depth - 1, alpha, beta, True) if depth > 1 else (self.evaluate(child), None)
                if score < value:
                    value, best_move = score, mv
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value, best_move

    # ===== 候选生成：严格按题述优先级 =====
    def _ordered_candidates(self, st: state) -> List[Coord]:
        self._ensure_graph(st)
        # 更新可能桥与真桥
        self.May_Hinger(st, full_scan=True)
        try:
            st.IS_Hinger(full_scan=True)
            true_h = list(st.Get_hinger_global_coords() or [])
        except Exception:
            true_h = []

        grid = st.result
        actives = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v > 0]
        gt2 = [(r, c) for (r, c) in actives if grid[r][c] > 2]
        eq2 = [(r, c) for (r, c) in actives if grid[r][c] == 2]
        eq1 = [(r, c) for (r, c) in actives if grid[r][c] == 1]

        # 第三层：值为2，且“不会产生任何可能桥”
        eq2_no_poss = [(r, c) for (r, c) in eq2 if self._move_creates_no_possible_bridges(st, r, c)]
        # 第四层：值为1，且“不会产生任何可能桥”
        eq1_no_poss = [(r, c) for (r, c) in eq1 if self._move_creates_no_possible_bridges(st, r, c)]
        # 第五层：值为2，且“产生的可能桥集中存在至少一个不是真桥”
        eq2_has_non_true = [(r, c) for (r, c) in eq2 if self._move_creates_non_true_possible_bridge(st, r, c)]
        # 第六层：值为1，且“产生的可能桥集中存在至少一个不是真桥”
        eq1_has_non_true = [(r, c) for (r, c) in eq1 if self._move_creates_non_true_possible_bridge(st, r, c)]

        ordered = []
        # 1) 真桥优先
        ordered += true_h
        # 2) >2
        ordered += gt2
        # 3) 2且不产可能桥
        ordered += eq2_no_poss
        # 4) 1且不产可能桥
        ordered += eq1_no_poss
        # 5) 2且存在非真“可能桥”
        ordered += eq2_has_non_true
        # 6) 1且存在非真“可能桥”
        ordered += eq1_has_non_true
        # 7) 其他活跃格兜底
        ordered += actives

        # 去重保序
        seen, res = set(), []
        for mv in ordered:
            if mv not in seen:
                seen.add(mv)
                res.append(mv)
        return res

    # ===== 工具：模拟一步后“不会产生任何可能桥” =====
    def _move_creates_no_possible_bridges(self, st: state, r: int, c: int) -> bool:
        child = self._clone_with_move(st, (r, c))
        self.May_Hinger(child, full_scan=True)
        poss = self._list_possible_bridge_globals_on(child)
        return len(poss) == 0

    # ===== 工具：模拟一步后“产生的可能桥集中是否存在不是真桥” =====
    def _move_creates_non_true_possible_bridge(self, st: state, r: int, c: int) -> bool:
        child = self._clone_with_move(st, (r, c))
        self.May_Hinger(child, full_scan=True)
        poss = set(self._list_possible_bridge_globals_on(child))
        try:
            child.IS_Hinger(full_scan=True)
            true_h = set(child.Get_hinger_global_coords() or [])
        except Exception:
            true_h = set()
        # 若存在“可能桥”不在“真桥”集合内 → 返回 True
        return any(p not in true_h for p in poss)

    # ===== 工具：克隆局面并在 (r,c) 处减一 =====
    def _clone_with_move(self, st: state, mv: Coord) -> state:
        r, c = mv
        grid = [row[:] for row in st.result]
        if 0 <= r < len(grid) and 0 <= c < (len(grid[0]) if grid else 0) and grid[r][c] > 0:
            grid[r][c] -= 1
        child = state(grid)
        self._ensure_graph(child)
        return child

    # ===== 工具：收集全局“可能桥”坐标（从节点 array_data 投影）=====
    def _list_possible_bridge_globals_on(self, st: state) -> List[Coord]:
        coords: List[Coord] = []
        cur = st.mylist.head if hasattr(st, "mylist") and st.mylist else None
        while cur is not None:
            arr = cur.get_array_data() or []
            rows = len(arr)
            cols = len(arr[0]) if rows > 0 else 0
            min_x, min_y = cur.get_min_x(), cur.get_min_y()
            for i in range(rows):
                for j in range(cols):
                    if arr[i][j] == 1:
                        gi = (min_y - 1) + i
                        gj = (min_x - 1) + j
                        # 仅收集仍为活跃格的坐标
                        if 0 <= gi < st.m and 0 <= gj < st.n and st.result[gi][gj] > 0:
                            coords.append((gi, gj))
            cur = cur.next
        return coords

    # ===== 工具：确保链表图已构建 =====
    def _ensure_graph(self, st: state) -> None:
        if hasattr(st, "Get_Graph"):
            try:
                st.Get_Graph()
            except Exception:
                pass

    def May_Hinger(self, state: 'a1_state.State' = None, node: Optional['ListNode'] = None,
                   full_scan: bool = False, return_new_bridge: bool = False) -> Optional[str]:
        """
        优化后的 May_Hinger 方法，完全在局部坐标空间中操作。
        关键修改：
        - 节点网格尺寸为 (rows+2) x (cols+2)，局部坐标范围 [0, rows+1] x [0, cols+1]
        - 所有检查在局部坐标中进行，避免不必要的全局坐标转换
        - 严格遵循"一圈空白"的节点结构
        """
        if state is None or state.mylist is None:
            return "false" if return_new_bridge else None

        # 全量扫描：处理所有节点
        if full_scan:
            current_node = state.mylist.head
            while current_node is not None:
                self._process_node_may_hinger(current_node)
                current_node = current_node.next
            return "false" if return_new_bridge else None

        # 增量更新：只处理指定节点
        elif node is not None:
            # 保存旧的候选桥梁坐标（局部坐标）
            old_array_data = node.get_array_data()
            old_bridge_coords: Set[Tuple[int, int]] = set()

            if old_array_data:
                rows = len(old_array_data)
                cols = len(old_array_data[0]) if rows > 0 else 0
                for i_local in range(rows):
                    for j_local in range(cols):
                        if old_array_data[i_local][j_local] == 1:
                            old_bridge_coords.add((i_local, j_local))

            # 处理当前节点，更新候选标记
            self._process_node_may_hinger(node)

            # 检测新桥梁
            new_array_data = node.get_array_data()
            new_bridge_coords: Set[Tuple[int, int]] = set()

            if new_array_data:
                rows = len(new_array_data)
                cols = len(new_array_data[0]) if rows > 0 else 0
                for i_local in range(rows):
                    for j_local in range(cols):
                        if new_array_data[i_local][j_local] == 1:
                            new_bridge_coords.add((i_local, j_local))

            has_new_bridge = len(new_bridge_coords - old_bridge_coords) > 0

            if return_new_bridge:
                return "true" if has_new_bridge else "false"

        return None

    def _process_node_may_hinger(self, node: 'ListNode') -> None:
        """
        处理单个节点的潜在桥梁检测，完全使用局部坐标。
        关键修改：
        - 适应节点网格尺寸 (rows+2) x (cols+2)
        - 局部坐标范围: i_local ∈ [0, rows+1], j_local ∈ [0, cols+1]
        - 空白圈对应局部坐标的边界（值为0），内部区域从(1,1)开始
        """
        # 获取节点网格数据（局部坐标，包含空白圈）
        node_grid = node.get_grid().data
        node_rows = len(node_grid)  # = max_y - min_y + 3 (原始rows+2)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # 初始化或清空 array_data
        array_data = node.get_array_data()
        if not array_data or len(array_data) != node_rows or len(array_data[0]) != node_cols:
            array_data = [[0] * node_cols for _ in range(node_rows)]

        # 清空旧标记
        for i in range(node_rows):
            for j in range(node_cols):
                array_data[i][j] = 0

        # 遍历所有局部坐标（包括空白圈）
        for i_local in range(node_rows):
            for j_local in range(node_cols):
                # 只检查计数器值为1的单元格（在局部网格中）
                if node_grid[i_local][j_local] != 1:
                    continue

                # 进行桥梁判断（完全在局部坐标中）
                if self._check_potential_hinger_local(node, i_local, j_local):
                    array_data[i_local][j_local] = 1

        # 更新节点的 array_data
        node.set_array_data(array_data)  # 使用新添加的 set_array_data 方法

    def _check_potential_hinger_local(self, node: 'ListNode', i_local: int, j_local: int) -> bool:
        """
        在局部坐标中检查单个位置是否为潜在桥梁。
        关键修改：
        - 完全基于节点局部网格（尺寸 rows+2 x cols+2）
        - 使用局部坐标进行所有邻居检查
        - 适应"一圈空白"的结构
        """
        # 获取节点网格数据
        node_grid = node.get_grid().data
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # 1. 冯诺依曼邻居检查（上下左右方向）
        von_neumann_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 左、右、上、下
        zero_count = 0

        for di, dj in von_neumann_dirs:
            ni, nj = i_local + di, j_local + dj
            # 检查邻居是否在局部网格范围内
            if 0 <= ni < node_rows and 0 <= nj < node_cols:
                if node_grid[ni][nj] == 0:  # 邻居为空白（值为0）
                    # 只统计同一行或同一列的空白
                    if (di == 0 and dj != 0) or (di != 0 and dj == 0):  # 同行或同列
                        zero_count += 1

        if zero_count < 2:
            return False

        # 2. 摩尔邻居数量检查（八个方向）
        moore_dirs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        active_neighbors = 0
        neighbor_positions = []

        for di, dj in moore_dirs:
            ni, nj = i_local + di, j_local + dj
            if 0 <= ni < node_rows and 0 <= nj < node_cols:
                if node_grid[ni][nj] >= 1:  # 活跃邻居（值≥1）
                    active_neighbors += 1
                    neighbor_positions.append((ni, nj))

        if active_neighbors <= 1:
            return False

        # 3. 邻居连通性检查（BFS在局部网格中）
        if len(neighbor_positions) == 0:
            return False

        visited = set()
        queue = collections.deque()
        start = neighbor_positions[0]
        queue.append(start)
        visited.add(start)

        while queue:
            r, c = queue.popleft()
            # 四连通检查（上下左右）
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if (neighbor in neighbor_positions and
                        neighbor not in visited and
                        0 <= nr < node_rows and 0 <= nc < node_cols):
                    visited.add(neighbor)
                    queue.append(neighbor)

        # 如果所有活跃邻居连通，则不是桥梁
        if len(visited) == len(neighbor_positions):
            return False

        return True


    def decision_tree(self, state, event_x: int, event_y: int, grid_row: int, grid_col: int):
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
            current_value = state.result[grid_row][grid_col]
            print(f"坐标({grid_row}, {grid_col})的计数器值: {current_value}")

            # 决策点1: 计数器为0或>2?
            if current_value == 0 or current_value > 2:
                print("决策点1: 计数器为0或大于2，结束处理")
                return

            # 决策点2: 计数器为1或2，调用change_data修改数据
            print("决策点2: 计数器为1或2，调用change_data修改数据")
            # 注意：这里需要传递node参数给Change_Data方法
            affected_node = state.Search_Node(grid_row, grid_col)
            if affected_node is None:
                print("警告: 未找到受影响节点，结束处理")
                return

            success = state.Change_Data(grid_row, grid_col, affected_node)  # 添加node参数
            if not success:
                print("数据修改失败，结束处理")
                return

            print(f"找到受影响节点: 范围({affected_node.get_min_x()},{affected_node.get_min_y()})到({affected_node.get_max_x()},{affected_node.get_max_y()})")

            # 决策点3: 调用May_Hinger → 返回new_bridge标志
            print("决策点3: 调用May_Hinger检查潜在桥梁")
            new_bridge = self.May_Hinger(state=state, node=affected_node, return_new_bridge=True)
            print(f"May_Hinger返回new_bridge = {new_bridge}")

            # 决策点4: new_bridge为"true"?
            if new_bridge != "true":
                print("决策点4: new_bridge不为'true'，结束处理")
                return

            print("决策点4: new_bridge为'true'，继续处理")

            # 决策点5: 调用IS_Hinger → 更新全局真桥列表
            print("决策点5: 调用IS_Hinger判断真桥")
            state.IS_Hinger(node=affected_node)

            # 更新桥梁数量
            state.numHingers()
            print(f"决策树处理完成，当前桥梁数量: {state.hinger_count}")

    def MCTS(self):
        pass
# 覆盖多类搜索与启发情形的测试集
TEST_CASES_MINI_ALPHA = [
    # 1) 十字真桥：中心为真桥（真桥应被优先选择）
    ("十字真桥", [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], 2, 3),

    # 2) 真桥 + 存在 >2：依然先点真桥而非 >2
    ("真桥优先于>2", [
        [0, 1, 0],
        [1, 3, 1],
        [0, 1, 0],
    ], 2, 3),

    # 3) 纯 >2 优先：无真桥时应优先点击 3
    ("仅>2优先", [
        [0, 1, 0],
        [1, 3, 1],
        [0, 1, 0],
        [0, 1, 0],
    ], 2, 3),

    # 4) 水平走廊（1x6）：任意中段通常为真桥
    ("水平走廊", [
        [1, 1, 1, 1, 1, 1],
    ], 2, 3),

    # 5) 垂直走廊（6x1）：任意中段通常为真桥
    ("垂直走廊", [
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
    ], 2, 3),

    # 6) 2 的“减一不产生可能桥”：致密块中点
    ("2减一不产可能桥(致密块)", [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ], 2, 3),

    # 7) 1 的“减一不产生可能桥”：边角安全点
    ("1减一不产可能桥(边角)", [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ], 2, 3),

    # 8) 2 的“产生可能桥但非真桥”：有替代连通路径
    ("2产可能桥但非真桥", [
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
    ], 2, 3),

    # 9) 1 的“产生可能桥但非真桥”：近似环绕
    ("1产可能桥但非真桥", [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0],
    ], 2, 3),

    # 10) 环形包围：移除单点通常不致断开（少真桥、少可能桥）
    ("环形(中空)", [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ], 3, 4),

    # 11) 棋盘格：8 邻域下连通性强，少真桥
    ("棋盘格", [
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
    ], 2, 3),

    # 12) 两个相距较远的分岛
    ("分离岛", [
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
    ], 2, 3),

    # 13) 多个 2 的十字：考察“2不产可能桥→1不产可能桥→产可能桥非真→其他”
    ("多2十字", [
        [0, 2, 0],
        [2, 2, 2],
        [0, 2, 0],
    ], 2, 3),

    # 14) T 字分叉：路口点多为真桥
    ("T字分叉", [
        [0, 1, 0, 0],
        [1, 2, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
    ], 2, 3),

    # 15) L 形走廊：拐角附近常出现真桥
    ("L形走廊", [
        [1, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
    ], 2, 3),

    # 16) 混合 >2 与稀疏 1：便于观察“>2 优先”与启发评分
    ("混合>2稀疏1", [
        [2, 0, 3, 0, 1],
        [0, 1, 0, 2, 0],
        [1, 0, 2, 0, 1],
    ], 2, 3),
]

def run_minimax_alphabeta_tests(cases=TEST_CASES_MINI_ALPHA):
    try:
        ag = Agent()
    except Exception:
        # 若当前作用域已有 Agent 定义则直接实例化
        ag = globals().get("Agent")()

    for idx, (name, grid, d_mm, d_ab) in enumerate(cases, 1):
        print(f"=== Case {idx}: {name} ===")
        st = state(grid)
        st.Get_Graph()

        # 运行 MiniMax
        mm_score, mm_move = None, None
        try:
            mm_score, mm_move = ag.MiniMax(st, depth=d_mm, maximizing_player=True)
            print(f"MiniMax(depth={d_mm}) -> score={mm_score}, move={mm_move}")
        except Exception as e:
            print(f"MiniMax 异常: {e}")

        # 运行 Alpha-Beta
        ab_score, ab_move = None, None
        try:
            ab_score, ab_move = ag.AlphaBeta(st, depth=d_ab, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
            print(f"AlphaBeta(depth={d_ab}) -> score={ab_score}, move={ab_move}")
        except Exception as e:
            print(f"AlphaBeta 异常: {e}")

        # 应用各自推荐一步查看落子效果
        def apply_and_show(move, tag):
            if move is None:
                return
            try:
                child = ag._clone_with_move(st, move)
                print(f"{tag} 推荐 {move} 后网格：")
                for r in child.result:
                    print(" ".join(f"{v:2d}" for v in r))
            except Exception as e:
                print(f"{tag} 应用落子异常: {e}")

        apply_and_show(mm_move, "MiniMax")
        apply_and_show(ab_move, "AlphaBeta")
        print()

if __name__ == "__main__":
    run_minimax_alphabeta_tests()