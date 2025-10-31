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

# If the ListNode is in another module, it needs to be imported
try:
    from MyList import ListNode
except ImportError:
    # If MyList is defined in the same file, use the global definition
    pass

Coord = Tuple[int, int]

class Agent:
    def __init__(self, state=None):
        self.name = "B20"
        self.size = "m,n"
        self.model = ["minimax", "alphabeta"]
        self.state = state # store reference to State instance

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

    # ===== Evaluation: more true hingers, more >2, fewer possible bridges, smaller total =====
    def evaluate(self, st: state) -> float:
        self._ensure_graph(st)
        # Refresh possible bridges and true bridges
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

    # ===== MiniMax: select first choice according to specified priorities =====
    def MiniMax(self, st: state, depth: int = 1, maximizing_player: bool = True):
        cands = self._ordered_candidates(st)
        best = cands[0] if cands else None
        return self.evaluate(st), best

    # ===== Alpha-Beta: expand and prune using the same candidate order =====
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

    # ===== Candidate generation: strict priority order as described =====
    def _ordered_candidates(self, st: state) -> List[Coord]:
        self._ensure_graph(st)
        # update possible and true hingers
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

        # Level 3: value == 2 and "creates no possible bridges"
        eq2_no_poss = [(r, c) for (r, c) in eq2 if self._move_creates_no_possible_bridges(st, r, c)]
        # Level 4: value == 1 and "creates no possible bridges"
        eq1_no_poss = [(r, c) for (r, c) in eq1 if self._move_creates_no_possible_bridges(st, r, c)]
        # Level 5: value == 2 and "there exists at least one possible bridge that is not a true bridge"
        eq2_has_non_true = [(r, c) for (r, c) in eq2 if self._move_creates_non_true_possible_bridge(st, r, c)]
        # Level 6: value == 1 and "there exists at least one possible bridge that is not a true bridge"
        eq1_has_non_true = [(r, c) for (r, c) in eq1 if self._move_creates_non_true_possible_bridge(st, r, c)]

        ordered = []
        # 1) true hingers first
        ordered += true_h
        # 2) >2
        ordered += gt2
        # 3) 2 and no possible bridges
        ordered += eq2_no_poss
        # 4) 1 and no possible bridges
        ordered += eq1_no_poss
        # 5) 2 and has a non-true possible bridge
        ordered += eq2_has_non_true
        # 6) 1 and has a non-true possible bridge
        ordered += eq1_has_non_true
        # 7) fallback: other active cells
        ordered += actives

        # remove duplicates while preserving order
        seen, res = set(), []
        for mv in ordered:
            if mv not in seen:
                seen.add(mv)
                res.append(mv)
        return res

    # ===== Utility: simulate one move and check "creates no possible bridges" =====
    def _move_creates_no_possible_bridges(self, st: state, r: int, c: int) -> bool:
        child = self._clone_with_move(st, (r, c))
        self.May_Hinger(child, full_scan=True)
        poss = self._list_possible_bridge_globals_on(child)
        return len(poss) == 0

    # ===== Utility: simulate one move and check "there exists a possible bridge that is not true" =====
    def _move_creates_non_true_possible_bridge(self, st: state, r: int, c: int) -> bool:
        child = self._clone_with_move(st, (r, c))
        self.May_Hinger(child, full_scan=True)
        poss = set(self._list_possible_bridge_globals_on(child))
        try:
            child.IS_Hinger(full_scan=True)
            true_h = set(child.Get_hinger_global_coords() or [])
        except Exception:
            true_h = set()
        # If there exists a possible bridge not in the true-bridge set -> return True
        return any(p not in true_h for p in poss)

    # ===== Utility: clone state and decrement at (r,c) =====
    def _clone_with_move(self, st: state, mv: Coord) -> state:
        r, c = mv
        grid = [row[:] for row in st.result]
        if 0 <= r < len(grid) and 0 <= c < (len(grid[0]) if grid else 0) and grid[r][c] > 0:
            grid[r][c] -= 1
        child = state(grid)
        self._ensure_graph(child)
        return child

    # ===== Utility: collect global possible bridge coordinates (projected from node array_data) =====
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
                        # only collect coordinates that are still active
                        if 0 <= gi < st.m and 0 <= gj < st.n and st.result[gi][gj] > 0:
                            coords.append((gi, gj))
            cur = cur.next
        return coords

    # ===== Utility: ensure linked-list graph has been built =====
    def _ensure_graph(self, st: state) -> None:
        if hasattr(st, "Get_Graph"):
            try:
                st.Get_Graph()
            except Exception:
                pass

    def May_Hinger(self, state: 'a1_state.State' = None, node: Optional['ListNode'] = None,
                   full_scan: bool = False, return_new_bridge: bool = False) -> Optional[str]:
        """
        Optimized May_Hinger method operating entirely in local coordinate space.
        Key changes:
        - Node grid size is (rows+2) x (cols+2), local coordinate range [0, rows+1] x [0, cols+1]
        - All checks operate in local coordinates to avoid unnecessary global conversions
        - Strictly follows the node structure with a one-cell blank border
        """
        if state is None or state.mylist is None:
            return "false" if return_new_bridge else None

        # Full scan: process all nodes
        if full_scan:
            current_node = state.mylist.head
            while current_node is not None:
                self._process_node_may_hinger(current_node)
                current_node = current_node.next
            return "false" if return_new_bridge else None

        # Incremental update: process only specified node
        elif node is not None:
            # Save old candidate bridge coordinates (local coordinates)
            old_array_data = node.get_array_data()
            old_bridge_coords: Set[Tuple[int, int]] = set()

            if old_array_data:
                rows = len(old_array_data)
                cols = len(old_array_data[0]) if rows > 0 else 0
                for i_local in range(rows):
                    for j_local in range(cols):
                        if old_array_data[i_local][j_local] == 1:
                            old_bridge_coords.add((i_local, j_local))

            # Process current node and update candidate markers
            self._process_node_may_hinger(node)

            # Detect new bridges
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
        Process a single node's potential bridge detection, using local coordinates.
        Key changes:
        - Adapt to node grid size (rows+2) x (cols+2)
        - Local coordinate range: i_local in [0, rows+1], j_local in [0, cols+1]
        - The blank border corresponds to local grid edges (value 0), interior starts at (1,1)
        """
        # Get node grid data (local coords, includes blank border)
        node_grid = node.get_grid().data
        node_rows = len(node_grid)  # = max_y - min_y + 3 (original rows+2)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # Initialize or clear array_data
        array_data = node.get_array_data()
        if not array_data or len(array_data) != node_rows or len(array_data[0]) != node_cols:
            array_data = [[0] * node_cols for _ in range(node_rows)]

        # Clear old markers
        for i in range(node_rows):
            for j in range(node_cols):
                array_data[i][j] = 0

        # Iterate all local coordinates (including border)
        for i_local in range(node_rows):
            for j_local in range(node_cols):
                # Only check cells with counter value 1 (in local grid)
                if node_grid[i_local][j_local] != 1:
                    continue

                # Perform bridge test (completely in local coordinates)
                if self._check_potential_hinger_local(node, i_local, j_local):
                    array_data[i_local][j_local] = 1

        # Update node's array_data
        node.set_array_data(array_data)  # use the newly added set_array_data method

    def _check_potential_hinger_local(self, node: 'ListNode', i_local: int, j_local: int) -> bool:
        """
        Check whether a single position is a potential bridge in local coordinates.
        Key changes:
        - Fully based on node local grid (size rows+2 x cols+2)
        - Use local coordinates for all neighbor checks
        - Adapt to the 'one-cell blank border' structure
        """
        # Get node grid data
        node_grid = node.get_grid().data
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # 1. Von Neumann neighbors check (up/down/left/right)
        von_neumann_dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # left, right, up, down
        zero_count = 0

        for di, dj in von_neumann_dirs:
            ni, nj = i_local + di, j_local + dj
            # Check neighbor is within local grid bounds
            if 0 <= ni < node_rows and 0 <= nj < node_cols:
                if node_grid[ni][nj] == 0:  # neighbor is blank (value 0)
                    # Only count blanks in same row or same column
                    if (di == 0 and dj != 0) or (di != 0 and dj == 0):  # same row or same column
                        zero_count += 1

        if zero_count < 2:
            return False

        # 2. Moore neighborhood check (8 directions)
        moore_dirs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if (di, dj) != (0, 0)]
        active_neighbors = 0
        neighbor_positions = []

        for di, dj in moore_dirs:
            ni, nj = i_local + di, j_local + dj
            if 0 <= ni < node_rows and 0 <= nj < node_cols:
                if node_grid[ni][nj] >= 1:  # active neighbor (value >= 1)
                    active_neighbors += 1
                    neighbor_positions.append((ni, nj))

        if active_neighbors <= 1:
            return False

        # 3. Neighbor connectivity check (BFS in local grid)
        if len(neighbor_positions) == 0:
            return False

        visited = set()
        queue = collections.deque()
        start = neighbor_positions[0]
        queue.append(start)
        visited.add(start)

        while queue:
            r, c = queue.popleft()
            # 4-connected check (up/down/left/right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if (neighbor in neighbor_positions and
                        neighbor not in visited and
                        0 <= nr < node_rows and 0 <= nc < node_cols):
                    visited.add(neighbor)
                    queue.append(neighbor)

        # If all active neighbors are connected, then it's not a bridge
        if len(visited) == len(neighbor_positions):
            return False

        return True


    def decision_tree(self, state, event_x: int, event_y: int, grid_row: int, grid_col: int):
            """
            Decision tree method: handle mouse click events and coordinate other functions
            :param event_x: mouse click pixel coordinate x
            :param event_y: mouse click pixel coordinate y
            :param grid_row: clicked grid row coordinate
            :param grid_col: clicked grid column coordinate
            :return: no return value, but internal state will be updated
            """
            print("Start decision tree processing")

            # Get counter value at clicked position
            current_value = state.result[grid_row][grid_col]
            print(f"Counter value at ({grid_row}, {grid_col}): {current_value}")

            # Decision point 1: is the counter 0 or >2?
            if current_value == 0 or current_value > 2:
                print("Decision 1: counter is 0 or >2, stop processing")
                return

            # Decision point 2: counter is 1 or 2, call Change_Data to modify data
            print("Decision 2: counter is 1 or 2, calling Change_Data to update data")
            # Note: need to pass node parameter to Change_Data method
            affected_node = state.Search_Node(grid_row, grid_col)
            if affected_node is None:
                print("Warning: affected node not found, stop processing")
                return

            success = state.Change_Data(grid_row, grid_col, affected_node)  # added node parameter
            if not success:
                print("Data modification failed, stop processing")
                return

            print(f"Found affected node: range({affected_node.get_min_x()},{affected_node.get_min_y()}) to ({affected_node.get_max_x()},{affected_node.get_max_y()})")

            # Decision point 3: call May_Hinger -> returns new_bridge flag
            print("Decision 3: calling May_Hinger to check potential bridges")
            new_bridge = self.May_Hinger(state=state, node=affected_node, return_new_bridge=True)
            print(f"May_Hinger returned new_bridge = {new_bridge}")

            # Decision point 4: is new_bridge equal to 'true'?
            if new_bridge != "true":
                print("Decision 4: new_bridge is not 'true', stop processing")
                return

            print("Decision 4: new_bridge is 'true', continue processing")

            # Decision point 5: call IS_Hinger -> update global true hinger list
            print("Decision 5: calling IS_Hinger to determine true hingers")
            state.IS_Hinger(node=affected_node)

            # Update hinger count
            state.numHingers()
            print(f"Decision tree processing complete, current hinger count: {state.hinger_count}")

    def MCTS(self):
        pass
# Test set covering multiple search and heuristic scenarios
TEST_CASES_MINI_ALPHA = [
    # 1) Cross true hinger: center should be selected as a true hinger (highest priority)
    ("Cross true hinger", [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], 2, 3),

    # 2) True hinger + >2 present: still pick true hinger instead of >2
    ("True hinger over >2", [
        [0, 1, 0],
        [1, 3, 1],
        [0, 1, 0],
    ], 2, 3),

    # 3) Pure >2 priority: with no true hinger pick 3
    ("Only >2 priority", [
        [0, 1, 0],
        [1, 3, 1],
        [0, 1, 0],
        [0, 1, 0],
    ], 2, 3),

    # 4) Horizontal corridor (1x6): middle usually a true hinger
    ("Horizontal corridor", [
        [1, 1, 1, 1, 1, 1],
    ], 2, 3),

    # 5) Vertical corridor (6x1): middle usually a true hinger
    ("Vertical corridor", [
        [1],
        [1],
        [1],
        [1],
        [1],
        [1],
    ], 2, 3),

    # 6) 2 that when decremented does not create possible bridges: dense block center
    ("2 decrement no possible bridge (dense block)", [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ], 2, 3),

    # 7) 1 that when decremented does not create possible bridges: safe corner
    ("1 decrement no possible bridge (corner safe)", [
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 0],
    ], 2, 3),

    # 8) 2 produces possible bridges but not true bridges: alternate paths exist
    ("2 produces possible but not true bridge", [
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
    ], 2, 3),

    # 9) 1 produces possible but not true bridges: near-surrounding
    ("1 produces possible but not true bridge", [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
        [0, 1, 0],
    ], 2, 3),

    # 10) Ring enclosure: removing a single cell usually doesn't disconnect (few true & possible bridges)
    ("Ring (hollow)", [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
    ], 3, 4),

    # 11) Checkerboard: strong connectivity under 8-neighborhood, few true bridges
    ("Checkerboard", [
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
    ], 2, 3),

    # 12) Two distant islands
    ("Separated islands", [
        [1, 1, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 1, 1],
    ], 2, 3),

    # 13) Multiple 2s cross: examines '2 no poss -> 1 no poss -> poss non-true -> others'
    ("Multiple 2s cross", [
        [0, 2, 0],
        [2, 2, 2],
        [0, 2, 0],
    ], 2, 3),

    # 14) T-junction: junction points often true hingers
    ("T junction", [
        [0, 1, 0, 0],
        [1, 2, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
    ], 2, 3),

    # 15) L-shaped corridor: corners often produce true hingers
    ("L-shaped corridor", [
        [1, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
    ], 2, 3),

    # 16) Mixed >2 and sparse 1s: observe '>2 priority' and heuristic scores
    ("Mixed >2 and sparse 1s", [
        [2, 0, 3, 0, 1],
        [0, 1, 0, 2, 0],
        [1, 0, 2, 0, 1],
    ], 2, 3),
]

def run_minimax_alphabeta_tests(cases=TEST_CASES_MINI_ALPHA):
    try:
        ag = Agent()
    except Exception:
        # If Agent already exists in the current scope instantiate directly
        ag = globals().get("Agent")()

    for idx, (name, grid, d_mm, d_ab) in enumerate(cases, 1):
        print(f"=== Case {idx}: {name} ===")
        st = state(grid)
        st.Get_Graph()

        # Run MiniMax
        mm_score, mm_move = None, None
        try:
            mm_score, mm_move = ag.MiniMax(st, depth=d_mm, maximizing_player=True)
            print(f"MiniMax(depth={d_mm}) -> score={mm_score}, move={mm_move}")
        except Exception as e:
            print(f"MiniMax exception: {e}")

        # Run Alpha-Beta
        ab_score, ab_move = None, None
        try:
            ab_score, ab_move = ag.AlphaBeta(st, depth=d_ab, alpha=float('-inf'), beta=float('inf'), maximizing_player=True)
            print(f"AlphaBeta(depth={d_ab}) -> score={ab_score}, move={ab_move}")
        except Exception as e:
            print(f"AlphaBeta exception: {e}")

        # Apply recommended moves to show resulting grids
        def apply_and_show(move, tag):
            if move is None:
                return
            try:
                child = ag._clone_with_move(st, move)
                print(f"{tag} recommended {move}, resulting grid:")
                for r in child.result:
                    print(" ".join(f"{v:2d}" for v in r))
            except Exception as e:
                print(f"{tag} apply move exception: {e}")

        apply_and_show(mm_move, "MiniMax")
        apply_and_show(ab_move, "AlphaBeta")
        print()

if __name__ == "__main__":
    run_minimax_alphabeta_tests()
