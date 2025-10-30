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
from MyList import MyList

# 保持对 MyList 的依赖

class State:
    # Core state class that holds the grid data and graph logic
    # 保存网格数据与图相关逻辑的核心类
    def __init__(self, data):
        self.m = len(data)
        self.n = len(data[0]) if self.m > 0 else 0
        self.result = [row[:] for row in data] # 全局网格数据

        from MyList import MyList
        self.mylist = MyList() # 储存活跃区域的链表
        self.node = self.mylist.append(
            0, 0, 0, 0,
            [[0] * (self.n + 2) for _ in range(self.m + 2)],
            bridge_num=0,
            graph_num=1
        )
        self.true_hinger_global_coords = []  # 全局真桥全局坐标列表
        self.hinger_count = 0  # 桥梁数量
        self.first_check = True  # 是否是第一次检查
        self.affected_node = set()  # 受影响的节点集合
        # 新增：用于记录鼠标事件（像素与网格坐标）
        self.mouse_events = []  # [{'x': int, 'y': int, 'row': int|None, 'col': int|None}, ...]

    def record_mouse(self, x: int, y: int, row: int | None = None, col: int | None = None) -> None:
        """
        记录一次鼠标事件
        :param x: 鼠标点击的像素坐标 x
        :param y: 鼠标点击的像素坐标 y
        :param row: 可选的网格坐标行号
        :param col: 可选的网格坐标列号
        """
        # 记录鼠标事件信息
        self.mouse_events.append({'x': x, 'y': y, 'row': row, 'col': col})

        # 如果提供了行列信息，找到对应的节点并标记为受影响
        if row is not None and col is not None:
            node = self.Search_Node(row, col)
            if node is not None:
                self.affected_node.add(node)
                # 同时修改数据
                self.Change_Data(row, col)
        return None  # 如果没有提供行列信息或未找到节点，返回None

    def Moves(self, row: int, col: int) -> bool:
        """
        修改指定鼠标点击坐标位置的数字，减一操作
        不对零进行操作，仅判断，由a4的win函数判断
        :param row: 全局网格地行坐标
        :param col: 全局网格的列坐标
        如果已经是零，则保持不变
        :return: 修改成功返回True，否则返回False
        只修改鼠标点击位置所在的区域节点内的数据（现交由Change_Date修改）
        """

        # 检查当前格内计数器数字是否为零
        if self.result[row][col] == 0:
            print(f"坐标({row}, {col})的值已经是零，不进行操作")
            return False

        # 通过鼠标坐标找到包含该坐标的节点
        node = self.Search_Node(row, col)
        if node is None:
            # 如果找不到对应节点，可能是点击了非活跃区域
            print(f"警告: 坐标({row}, {col})不在任何活跃区域内")
            return False

    def Change_Data(self, row: int, col: int, node) -> bool:
        """
        修改受影响节点的指定坐标位置的数字，减一操作
        :param row: 行坐标（全局）
        :param col: 列坐标（全局）
        :param node: 包含该坐标的节点
        :return: 修改成功返回True，否则返回False
        只修改鼠标点击位置所在的区域节点内的数据
        """
        # 获取节点的边界信息
        min_x, max_x = node.get_min_x(), node.get_max_x()  # 列的范围
        min_y, max_y = node.get_min_y(), node.get_max_y()  # 行的范围

        # 获取节点的网格数据
        node_grid = node.get_grid().data

        # 计算节点网格的尺寸（包含周围一圈空白）
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # 将全局坐标转换为节点局部坐标
        # 节点网格的(0,0)对应全局的(min_y-1, min_x-1)
        local_row = row - (min_y - 1)
        local_col = col - (min_x - 1)

        # 确保局部坐标在节点网格范围内
        if (0 <= local_row < node_rows and
                0 <= local_col < node_cols):

            # 获取当前值
            current_value = node_grid[local_row][local_col]

            # 如果当前值大于零，则减一
            if current_value > 0:
                # 更新节点网格数据
                node_grid[local_row][local_col] = current_value - 1

                # 同时更新全局网格数据
                self.result[row][col] = current_value - 1

                print(f"坐标({row}, {col})的值从{current_value}减少到{current_value - 1}")
                return True
            else:
                print(f"坐标({row}, {col})的值已经是零，保持不变")
                return False
        else:
            print(f"错误: 局部坐标({local_row}, {local_col})超出节点网格范围")
            print(f"节点边界: min_y={min_y}, max_y={max_y}, min_x={min_x}, max_x={max_x}")
            print(f"节点网格尺寸: {node_rows} x {node_cols}")
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

                # 队列与起点
                arr_ones = [(i, j)]
                head = 0

                # 边界与采样
                minrow = maxrow = i
                mincol = maxcol = j
                comp_cells = []  # (r, c, v)

                # 起点入采样，写入临时 grid（不清零 graph）
                v0 = graph[i][j]
                grid[i][j] = v0
                visited[i][j] = True
                comp_cells.append((i, j, v0))

                # BFS
                while head < len(arr_ones):
                    r, c = arr_ones[head]
                    head += 1

                    # 更新边界
                    if r < minrow: minrow = r
                    if r > maxrow: maxrow = r
                    if c < mincol: mincol = c
                    if c > maxcol: maxcol = c

                    # 扩展八邻域
                    for dr, dc in dirs8:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.m and 0 <= nc < self.n:
                            if not visited[nr][nc] and graph[nr][nc] != 0:
                                val = graph[nr][nc]
                                arr_ones.append((nr, nc))
                                grid[nr][nc] = val
                                visited[nr][nc] = True
                                comp_cells.append((nr, nc, val))

                                # 同步更新边界
                                if nr < minrow: minrow = nr
                                if nr > maxrow: maxrow = nr
                                if nc < mincol: mincol = nc
                                if nc > maxcol: maxcol = nc

                # 子网络裁剪：最小行/列对齐到 (1,1)，尺寸为(最大行+1, 最大列+1)
                rows_rel_max = (maxrow - minrow) + 1
                cols_rel_max = (maxcol - mincol) + 1
                subgrid = [[0] * (cols_rel_max + 1) for _ in range(rows_rel_max + 1)]
                for r, c, v in comp_cells:
                    rr = (r - minrow) + 1
                    cc = (c - mincol) + 1
                    subgrid[rr][cc] = v

                # 写入链表：头节点直写，其后尾插
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
                    # 检查是否存在相同图形（比较边界与二维数组内容）
                    is_dup = False
                    for node in self.mylist:
                        if (node.get_min_x() == mincol and node.get_max_x() == maxcol and
                                node.get_min_y() == minrow and node.get_max_y() == maxrow and
                                node.get_grid().data == subgrid):
                            is_dup = True
                            break

                    # 不重复则尾插，重复则跳过
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

                # 清零本地缓冲 grid（不影响 self.result）
                for rr in range(self.m):
                    for cc in range(self.n):
                        grid[rr][cc] = 0

    def Search_Node(self, row, col):
        """
        根据坐标找到对应的链表节点
        :param row: 全局网格的行坐标
        :param col: 全局网格的列坐标
        :return: 对应的链表节点或 None
        每个节点代表一个活跃区域，有特定的坐标范围
        """

        """
        # 全新优化（by HaoranXiong）首先检查最近访问的节点（缓存优化）
        if hasattr(self, 'last_accessed_node') and self.last_accessed_node:
            min_x, max_x = self.last_accessed_node.get_min_x(), self.last_accessed_node.get_max_x()
            min_y, max_y = self.last_accessed_node.get_min_y(), self.last_accessed_node.get_max_y()
            if (min_y - 1 <= row <= max_y + 1) and (min_x - 1 <= col <= max_x + 1):
                if self.result[row][col] > 0:
                    return self.last_accessed_node
        """

        current_node = self.mylist.head
        while current_node is not None:
            # 检查坐标是否在节点的活跃坐标集合中
            if hasattr(current_node, 'get_active_coords_set') and (row, col) in current_node.get_active_coords_set():
                return current_node
            # 备用方法：如果节点没有活跃坐标集合，使用边界检查
            elif hasattr(current_node, 'get_min_x'):
                min_x, max_x = current_node.get_min_x(), current_node.get_max_x()
                min_y, max_y = current_node.get_min_y(), current_node.get_max_y()
                if min_x <= row <= max_x and min_y <= col <= max_y:
                    # 进一步检查坐标是否确实活跃
                    if self.result[row][col] > 0:
                        return current_node
            current_node = current_node.next
        return None

    def IS_Hinger(self, node=None, full_scan=False):
        """
        判断真正桥梁并更新全局真桥坐标列表
        所有检查操作完全在节点局部坐标中进行，仅在确认后转换为全局坐标
        :param node: 可选，指定要检查的节点（用于增量更新）
        :param full_scan: 是否进行全量扫描（首次调用时设置为True）
        :return: 更新后的全局真桥坐标列表
        """
        if full_scan:
            # 首次调用：遍历所有节点(全量扫描)
            self.true_hinger_global_coords = []
            current_node = self.mylist.head
            while current_node is not None:
                self._process_node_true_hingers_local(current_node)
                current_node = current_node.next
        elif node is not None:
            # 增量更新：只处理指定节点，移除节点范围内的旧桥梁，添加新桥梁
            self._remove_node_hingers_from_global(node)
            # 然后处理该节点中的新的可能的桥梁
            self._process_node_true_hingers_local(node)

        # 更新桥梁数量
        self.hinger_count = len(self.true_hinger_global_coords)
        return self.true_hinger_global_coords

    def _process_node_true_hingers_local(self, node):
        """
        在局部坐标空间中处理单个节点的真正桥梁判断
        所有操作使用节点局部坐标，避免任何越界检查
        在局部坐标空间中处理真桥检测，确认后再转换全局坐标
        """
        # 获取节点边界信息（用于最后的坐标转换）
        min_x, max_x = node.get_min_x(), node.get_max_x()
        min_y, max_y = node.get_min_y(), node.get_max_y()

        # 获取节点的网格数据（局部坐标）
        node_grid = node.get_grid().data
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # 获取May_Hinger标记的潜在桥梁（局部坐标）
        array_data = node.get_array_data()

        # 遍历节点中的潜在桥梁位置（使用局部坐标）
        for i_local in range(node_rows):
            for j_local in range(node_cols):
                if array_data[i_local][j_local] == 1:  # 潜在桥梁位置
                    # 在局部坐标空间中检查是否为真正桥梁
                    if self._check_hinger_creates_new_region_local(node, i_local, j_local):
                        # 如果是真正桥梁，将局部坐标转换为全局坐标
                        i_global = min_y - 1 + i_local
                        j_global = min_x - 1 + j_local

                        # 确保全局坐标有效
                        if (0 <= i_global < self.m and 0 <= j_global < self.n):
                            # 添加到全局真桥列表
                            if (i_global, j_global) not in self.true_hinger_global_coords:
                                self.true_hinger_global_coords.append((i_global, j_global))

    def _check_hinger_creates_new_region_local(self, node, i_local, j_local):
        """
        在局部坐标空间中检查移除指定桥梁是否会产生新的活跃区域
        使用BFS算法，完全在节点局部网格内操作
        """
        # 获取节点的局部网格数据
        node_grid = node.get_grid().data
        node_rows = len(node_grid)
        node_cols = len(node_grid[0]) if node_rows > 0 else 0

        # 创建网格副本用于模拟操作
        grid_copy = [row[:] for row in node_grid]

        # 模拟移除该桥梁（在局部网格中临时设置为0）
        original_value = grid_copy[i_local][j_local]
        grid_copy[i_local][j_local] = 0

        # 使用BFS检查连通组件数量（在局部坐标中）
        visited = set()
        region_count = 0

        # 获取节点局部网格中的所有活跃单元格
        active_cells = set()
        for i in range(node_rows):
            for j in range(node_cols):
                if grid_copy[i][j] > 0:
                    active_cells.add((i, j))

        """基本无意义
        # 如果没有活跃单元格，不会产生新区域
        if not active_cells:
            return False
        """

        # BFS遍历
        for cell in active_cells:
            if cell not in visited:
                region_count += 1
                if region_count > 1:
                    # 如果已经找到多于一个区域，提前终止
                    break

                # BFS遍历当前区域
                queue = collections.deque([cell])
                visited.add(cell)

                while queue:
                    r, c = queue.popleft()
                    # 检查八个方向（摩尔邻居）
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            neighbor = (nr, nc)

                            # 确保邻居在节点网格范围内且是活跃单元格
                            if (0 <= nr < node_rows and 0 <= nc < node_cols and
                                    grid_copy[nr][nc] > 0 and neighbor not in visited):
                                visited.add(neighbor)
                                queue.append(neighbor)

        # 如果区域数量大于1，表示会产生新区域
        return region_count > 1

    def _remove_node_hingers_from_global(self, node):
        """
        从全局真桥列表中移除指定节点范围内的所有桥梁
        基于节点边界信息进行过滤
        """
        min_x, max_x = node.get_min_x(), node.get_max_x()
        min_y, max_y = node.get_min_y(), node.get_max_y()

        # 计算节点在全局网格中的实际范围
        global_min_row = min_y - 1
        global_max_row = max_y + 1
        global_min_col = min_x - 1
        global_max_col = max_x + 1

        # 过滤掉在节点范围内的坐标
        self.true_hinger_global_coords = [
            coord for coord in self.true_hinger_global_coords
            if not (global_min_row <= coord[0] <= global_max_row and
                    global_min_col <= coord[1] <= global_max_col)
        ]

    def numHingers(self):
        """
        直接读取全局真桥坐标列表的长度来统计桥梁数量
        更新self.hinger_count并返回当前桥梁数量
        """
        # 获取全局真桥坐标列表的长度
        self.hinger_count = len(self.true_hinger_global_coords)

        # 返回桥梁数量
        return self.hinger_count

    def numRegions(self):
        # Placeholder for returning number of regions
        # 返回活跃区域数量的占位函数
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
            # 新增：将鼠标像素坐标与网格坐标传递给 State
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
    # 程序入口
    tester()


