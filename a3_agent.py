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

import a1_state
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

    # ===== May_Hinger：为各节点生成 array_data（可能桥）=====
    def May_Hinger(self, st: state, node=None, full_scan: bool = False, return_new_bridge: bool = False):
        # 确保图结构可用
        self._ensure_graph(st)
        def compute_for_node(nd) -> bool:
            grid = nd.get_grid().data
            rows = len(grid)
            cols = len(grid[0]) if rows > 0 else 0
            old = nd.get_array_data()
            arr = [[0] * cols for _ in range(rows)]
            changed_new_one = False
            for i in range(rows):
                for j in range(cols):
                    if grid[i][j] > 0:
                        # 若移除此点会产生多个活跃区域 → 这是“可能桥”
                        try:
                            is_bridge_like = st._check_hinger_creates_new_region_local(nd, i, j)
                        except Exception:
                            is_bridge_like = False
                        arr[i][j] = 1 if is_bridge_like else 0
                        if return_new_bridge and arr[i][j] == 1:
                            if not (0 <= i < len(old) and 0 <= j < (len(old[0]) if old else 0) and old[i][j] == 1):
                                changed_new_one = True
            nd.array_data = arr
            return changed_new_one

        any_new = False
        if full_scan:
            cur = st.mylist.head if hasattr(st, "mylist") and st.mylist else None
            while cur is not None:
                if compute_for_node(cur):
                    any_new = True
                cur = cur.next
        elif node is not None:
            any_new = compute_for_node(node)

        if return_new_bridge:
            return "true" if any_new else "false"
        return None

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
# python
from a1_state import State as state
try:
    # 若与 Agent 在同一文件，可直接 from 当前模块导入或直接使用 Agent
    from a3_agent import Agent
except Exception:
    # 如果就在 a3_agent.py 里粘贴本段，确保上方已定义 Agent
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