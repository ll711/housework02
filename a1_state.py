#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086 and 100331080)
@date: 11/10/2025

"""
import tkinter as tk
import collections
from typing import Any

from MyList import MyList

# Keep dependency on MyList

class State:
    # Core state class that holds the grid data and graph logic
    def __init__(self, data):
        self.m = len(data)
        self.n = len(data[0]) if self.m > 0 else 0
        self.result = [row[:] for row in data]  # global grid data

        from MyList import MyList
        self.mylist = MyList()  # stores linked list of active regions
        self.node = self.mylist.append(
            0, 0, 0, 0,
            [[0] * (self.n + 2) for _ in range(self.m + 2)],
            bridge_num=0,
            graph_num=1
        )
        self.true_hinger_global_coords = []  # global true hinger coordinates list
        self.hinger_count = 0  # number of hingers
        self.first_check = True  # whether it's the first check
        self.affected_node = set()  # set of affected nodes
        # New: record mouse events (pixel and grid coordinates)
        self.mouse_events = []  # [{'x': int, 'y': int, 'row': int|None, 'col': int|None}, ...]

    def Get_hinger_global_coords(self):
        """
        Get the current list of true hinger coordinates
        :return: list of true hinger coordinates
        """
        return self.true_hinger_global_coords

    def get_result(self):
        """
        Get the current global grid data
        :return: global grid data
        """
        return self.result

    def getmylist(self):
        """
        Get the current linked-list object
        :return: linked-list object
        """
        return self.mylist

    def record_mouse(self, x: int, y: int, row: int | None = None, col: int | None = None) -> None:
        """
        Record a mouse event
        :param x: mouse click pixel coordinate x
        :param y: mouse click pixel coordinate y
        :param row: optional grid row index
        :param col: optional grid column index
        """
        # Record mouse event details
        self.mouse_events.append({'x': x, 'y': y, 'row': row, 'col': col})

        # If row/col provided, find the corresponding node and mark it affected
        if row is not None and col is not None:
            node = self.Search_Node(row, col)
            if node is not None:
                self.affected_node.add(node)
                # Also update data
                self.Change_Data(row, col)
        return None  # return None if row/col not provided or node not found

    def Moves(self, row: int, col: int) -> bool:
        """
        Decrement the counter at the specified mouse click coordinates.
        Do not operate on zeros; the GUI (a4) will handle display.
        :param row: global grid row index
        :param col: global grid column index
        If already zero, keep unchanged
        :return: True if modification succeeded, otherwise False
        Only modifies data inside the node that contains the clicked cell (Change_Data handles this)
        """

        # Check if the counter at the cell is zero
        if self.result[row][col] == 0:
            print(f"Value at ({row}, {col}) is already zero, no operation performed")
            return False

        # Find the node that contains the mouse coordinates
        node = self.Search_Node(row, col)
        if node is None:
            # If no node found, the click may have been outside active regions
            print(f"Warning: Coordinate ({row}, {col}) is not inside any active region")
            return False

    def Change_Data(self, row: int, col: int, node) -> bool:
        """
        Decrement the value at the specified global coordinate inside the affected node.
        :param row: row coordinate (global)
        :param col: column coordinate (global)
        :param node: node that contains the coordinate
        :return: True if modification succeeded, otherwise False
        Only modifies the data inside the node that contains the clicked cell
        """
        # Get node boundary information
        min_x, max_x = node.get_min_x(), node.get_max_x()  # column range
        min_y, max_y = node.get_min_y(), node.get_max_y()  # row range

        # Get the node's grid data
        node_grid = node.get_grid().data

        # Compute node grid size (includes surrounding blank border)
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # Convert global coordinates to local node coordinates
        # Node grid (0,0) corresponds to global (min_y-1, min_x-1)
        local_row = row - (min_y - 1)
        local_col = col - (min_x - 1)

        # Ensure local coordinates are within node grid bounds
        if (0 <= local_row < node_rows and
                0 <= local_col < node_cols):

            # Get current value
            current_value = node_grid[local_row][local_col]

            # If current value > 0, decrement it
            if current_value > 0:
                # Update node grid data
                node_grid[local_row][local_col] = current_value - 1

                # Also update global grid data
                self.result[row][col] = current_value - 1

                print(f"Value at ({row}, {col}) decreased from {current_value} to {current_value - 1}")
                return True
            else:
                print(f"Value at ({row}, {col}) is already zero, unchanged")
                return False
        else:
            print(f"Error: local coordinates ({local_row}, {local_col}) out of node grid bounds")
            print(f"Node boundaries: min_y={min_y}, max_y={max_y}, min_x={min_x}, max_x={max_x}")
            print(f"Node grid size: {node_rows} x {node_cols}")
            return False

    def Get_Graph(self):
        graph = self.result
        grid = [[0] * self.n for _ in range(self.m)]
        visited = [[False] * self.n for _ in range(self.m)]

        dirs8 = [
            (-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]

        used_head = False

        for i in range(self.m):
            for j in range(self.n):
                if graph[i][j] == 0 or visited[i][j]:
                    continue

                # queue and start point
                arr_ones = [(i, j)]
                head = 0

                # boundaries and sampling
                minrow = maxrow = i
                mincol = maxcol = j
                comp_cells = []  # (r, c, v)

                # add start point to sample, write into temporary grid (do not zero out graph)
                v0 = graph[i][j]
                grid[i][j] = v0
                visited[i][j] = True
                comp_cells.append((i, j, v0))

                # BFS
                while head < len(arr_ones):
                    r, c = arr_ones[head]
                    head += 1

                    # update boundaries
                    if r < minrow: minrow = r
                    if r > maxrow: maxrow = r
                    if c < mincol: mincol = c
                    if c > maxcol: maxcol = c

                    # expand 8-neighborhood
                    for dr, dc in dirs8:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.m and 0 <= nc < self.n:
                            if not visited[nr][nc] and graph[nr][nc] != 0:
                                val = graph[nr][nc]
                                arr_ones.append((nr, nc))
                                grid[nr][nc] = val
                                visited[nr][nc] = True
                                comp_cells.append((nr, nc, val))

                                # sync update boundaries
                                if nr < minrow: minrow = nr
                                if nr > maxrow: maxrow = nr
                                if nc < mincol: mincol = nc
                                if nc > maxcol: maxcol = nc

                # clip subnetwork: align minimal row/col to (1,1), size = (maxrow-minrow+1, maxcol-mincol+1)
                rows_rel_max = (maxrow - minrow) + 1
                cols_rel_max = (maxcol - mincol) + 1
                subgrid = [[0] * (cols_rel_max + 1) for _ in range(rows_rel_max + 1)]
                for r, c, v in comp_cells:
                    rr = (r - minrow) + 1
                    cc = (c - mincol) + 1
                    subgrid[rr][cc] = v

                # write into linked list: write head node directly, then append others
                if not used_head and self.mylist.head is not None and self.mylist.head == self.mylist.tail:
                    head_node = self.mylist.head
                    head_node.set_min_x(mincol)
                    head_node.set_max_x(maxcol)
                    head_node.set_min_y(minrow)
                    head_node.set_max_y(maxrow)
                    head_node.set_grid_data(subgrid)
                    head_node.set_graph_num(1)
                    used_head = True
                else:
                    # check for duplicate shapes (compare boundaries and 2D array content)
                    is_dup = False
                    for node in self.mylist:
                        if (node.get_min_x() == mincol and node.get_max_x() == maxcol and
                                node.get_min_y() == minrow and node.get_max_y() == maxrow and
                                node.get_grid().data == subgrid):
                            is_dup = True
                            break

                    # if not duplicate append, otherwise skip
                    if not is_dup:
                        last_graph_num = self.mylist.tail.get_graph_num() if self.mylist.tail else 0
                        self.mylist.append(
                            mincol, maxcol, minrow, maxrow,
                            subgrid, bridge_num=0, graph_num=last_graph_num + 1
                        )
                    last_graph_num = self.mylist.tail.get_graph_num() if self.mylist.tail else 0
                    self.mylist.append(
                        mincol, maxcol, minrow, maxrow,
                        subgrid, bridge_num=0, graph_num=last_graph_num + 1
                    )

                # clear local buffer grid (do not affect self.result)
                for rr in range(self.m):
                    for cc in range(self.n):
                        grid[rr][cc] = 0

    def Search_Node(self, row, col):
        """
        Find the linked-list node that contains the given coordinate
        :param row: global grid row index
        :param col: global grid column index
        :return: corresponding linked-list node or None
        Each node represents an active region with specific coordinate bounds
        """

        """
        # New optimization (by HaoranXiong): first check the last accessed node (cache optimization)
        if hasattr(self, 'last_accessed_node') and self.last_accessed_node:
            min_x, max_x = self.last_accessed_node.get_min_x(), self.last_accessed_node.get_max_x()
            min_y, max_y = self.last_accessed_node.get_min_y(), self.last_accessed_node.get_max_y()
            if (min_y - 1 <= row <= max_y + 1) and (min_x - 1 <= col <= max_x + 1):
                if self.result[row][col] > 0:
                    return self.last_accessed_node
        """

        current_node = self.mylist.head
        while current_node is not None:
            # Check if coordinate is in node's active coordinate set
            if hasattr(current_node, 'get_active_coords_set') and (row, col) in current_node.get_active_coords_set():
                return current_node
            # Fallback: if node doesn't have active coords set, use boundary check
            elif hasattr(current_node, 'get_min_x'):
                min_x, max_x = current_node.get_min_x(), current_node.get_max_x()
                min_y, max_y = current_node.get_min_y(), current_node.get_max_y()
                if min_x <= row <= max_x and min_y <= col <= max_y:
                    # further verify that the coordinate is actually active
                    if self.result[row][col] > 0:
                        return current_node
            current_node = current_node.next
        return None

    def IS_Hinger(self, node=None, full_scan=False):
        """
        Determine true hingers and update the global true hinger coordinate list.
        All checks operate entirely in node local coordinates and convert to global only upon confirmation.
        :param node: optional, specify a node to check (for incremental update)
        :param full_scan: whether to perform a full scan (set to True on first call)
        :return: updated global true hinger coordinates list
        """
        if full_scan:
            # First call: iterate all nodes (full scan)
            self.true_hinger_global_coords = []
            current_node = self.mylist.head
            while current_node is not None:
                self._process_node_true_hingers_local(current_node)
                current_node = current_node.next
        elif node is not None:
            # Incremental update: process only specified node, remove old hingers in the node range, add new ones
            self._remove_node_hingers_from_global(node)
            # then process new possible hingers in that node
            self._process_node_true_hingers_local(node)

        # Update hinger count
        self.hinger_count = len(self.true_hinger_global_coords)
        return self.true_hinger_global_coords

    def _process_node_true_hingers_local(self, node):
        """
        Process true hinger detection for a single node in local coordinate space.
        All operations use node local coordinates and avoid any unnecessary global conversions.
        """
        # Get node boundary info (used for final coordinate conversion)
        min_x, max_x = node.get_min_x(), node.get_max_x()
        min_y, max_y = node.get_min_y(), node.get_max_y()

        # Get the node's grid data (local coordinates)
        node_grid = node.get_grid().data
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # Get May_Hinger marked potential hingers (local coordinates)
        array_data = node.get_array_data()

        # Iterate potential hinger positions inside the node (local coordinates)
        for i_local in range(node_rows):
            for j_local in range(node_cols):
                if array_data[i_local][j_local] == 1:  # potential hinger position
                    # Check in local coordinate space whether it's a true hinger
                    if self._check_hinger_creates_new_region_local(node, i_local, j_local):
                        # If it's a true hinger, convert local to global coordinates
                        i_global = min_y - 1 + i_local
                        j_global = min_x - 1 + j_local

                        # Ensure global coordinates are valid
                        if (0 <= i_global < self.m and 0 <= j_global < self.n):
                            # Add to global true hinger list if not already present
                            if (i_global, j_global) not in self.true_hinger_global_coords:
                                self.true_hinger_global_coords.append((i_global, j_global))

    def _check_hinger_creates_new_region_local(self, node, i_local, j_local):
        """
        In local coordinate space, check whether removing the specified potential hinger
        would create a new active region. Uses BFS and operates entirely on the node local grid.
        """
        # Get the node's local grid data
        node_grid = node.get_grid().data
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # Create a copy of the grid for simulation
        grid_copy = [row[:] for row in node_grid]

        # Simulate removing that hinger (temporarily set to 0 in local grid)
        original_value = grid_copy[i_local][j_local]
        grid_copy[i_local][j_local] = 0

        # Use BFS to check the number of connected components (in local coords)
        visited = set()
        region_count = 0

        # Collect all active cells in the node local grid
        active_cells = set()
        for i in range(node_rows):
            for j in range(node_cols):
                if grid_copy[i][j] > 0:
                    active_cells.add((i, j))

        # BFS traversal
        for cell in active_cells:
            if cell not in visited:
                region_count += 1
                if region_count > 1:
                    # early exit if more than one region is found
                    break

                # BFS traverse current region
                queue = collections.deque([cell])
                visited.add(cell)

                while queue:
                    r, c = queue.popleft()
                    # check 8 directions (Moore neighborhood)
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            neighbor = (nr, nc)

                            # ensure neighbor is within node grid and is active
                            if (0 <= nr < node_rows and 0 <= nc < node_cols and
                                    grid_copy[nr][nc] > 0 and neighbor not in visited):
                                visited.add(neighbor)
                                queue.append(neighbor)

        # If number of regions > 1, removal creates a new region
        return region_count > 1

    def _remove_node_hingers_from_global(self, node):
        """
        Remove all hingers that lie within the specified node range from the global true hinger list.
        Uses node boundary information for filtering.
        """
        min_x, max_x = node.get_min_x(), node.get_max_x()
        min_y, max_y = node.get_min_y(), node.get_max_y()

        # Compute node's actual range in global grid
        global_min_row = min_y - 1
        global_max_row = max_y + 1
        global_min_col = min_x - 1
        global_max_col = max_x + 1

        # Filter out coordinates that fall inside the node range
        self.true_hinger_global_coords = [
            coord for coord in self.true_hinger_global_coords
            if not (global_min_row <= coord[0] <= global_max_row and
                    global_min_col <= coord[1] <= global_max_col)
        ]

    def numHingers(self):
        """
        Count the number of hingers by reading the global true hinger list length.
        Updates self.hinger_count and returns the current count.
        """
        # Update hinger count from global list length
        self.hinger_count = len(self.true_hinger_global_coords)

        # Return number of hingers
        return self.hinger_count

    def numRegions(self):
        # Placeholder for returning number of regions
        # Returns the number of active regions
        return len(self.mylist)

def tester():

    m = int(input("enter rows m: "))
    n = int(input("enter columns n: "))

    cell_size = min(60, 800 // max(m, n)) if m and n else 60
    pad = cell_size // 2
    canvas_width = n * cell_size + 2 * pad
    canvas_height = m * cell_size + 2 * pad

    root = tk.Tk()
    root.title("game")
    root.geometry(f"{canvas_width}x{canvas_height}")

    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
    canvas.pack()

    for i in range(m + 1):
        y = pad + i * cell_size
        canvas.create_line(pad, y, pad + n * cell_size, y)
    for j in range(n + 1):
        x = pad + j * cell_size
        canvas.create_line(x, pad, x, pad + m * cell_size)

    entries = []
    for i in range(m):
        row_entries = []
        for j in range(n):
            entry = tk.Entry(root, width=5, justify='center')
            x = pad + j * cell_size
            y = pad + i * cell_size
            entry.place(x=x + 2, y=y + 2, width=cell_size - 4, height=cell_size - 4)
            row_entries.append(entry)
        entries.append(row_entries)

    data = [[0] * n for _ in range(m)]
    game_started = {'value': False}
    state_holder = {'state': None}

    def on_enter(event=None):
        from a1_state import State
        for i in range(m):
            for j in range(n):
                val = entries[i][j].get()
                try:
                    num = int(val)
                except ValueError:
                    num = 0
                data[i][j] = num
                entries[i][j].destroy()
                cx = pad + j * cell_size + cell_size / 2
                cy = pad + i * cell_size + cell_size / 2
                canvas.create_text(
                    cx, cy, text=str(num), tags=f"cell_{i}_{j}",
                    font=("Arial", int(cell_size // 2))
                )
        game_started['value'] = True
        state_holder['state'] = State(data)
        state_holder['state'].Get_Graph()

    def on_click(event):
        if not game_started['value']:
            return
        gx = event.x - pad
        gy = event.y - pad
        if gx < 0 or gy < 0:
            return
        col = int(gx // cell_size)
        row = int(gy // cell_size)
        if 0 <= row < m and 0 <= col < n:
            # New: pass mouse pixel coordinates and grid coordinates to State
            if state_holder['state'] is not None:
                state_holder['state'].record_mouse(event.x, event.y, row=row, col=col)

            if data[row][col] > 0:
                data[row][col] -= 1
                canvas.delete(f"cell_{row}_{col}")
                cx = pad + col * cell_size + cell_size / 2
                cy = pad + row * cell_size + cell_size / 2
                canvas.create_text(
                    cx, cy, text=str(data[row][col]), tags=f"cell_{row}_{col}",
                    font=("Arial", int(cell_size // 2))
                )

    root.bind('<Return>', on_enter)
    canvas.bind('<Button-1>', on_click)
    root.mainloop()

if __name__ == "__main__":
    # Program entry point
    tester()


