"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086)
@date: 11/10/2025

"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Iterator

__all__ = ["Grid2D", "ListNode", "MyList"]

class Grid2D:
    # Lightweight 2D grid wrapper with bounds checking and shape helpers
    # 轻量级二维网格封装，提供边界检查与形状辅助方法
    def __init__(self, data: List[List[int]]):
        # Enforce equal-length rows to keep a proper rectangle grid
        # 强制每行等长以保持规则矩形网格
        if data and any(len(row) != len(data[0]) for row in data):
            raise ValueError("Grid2D rows must have equal length")
        self.data = data or []

    @property
    def shape(self) -> Tuple[int, int]:
        # Return grid dimensions as (rows, cols)
        # 返回网格尺寸 (行数, 列数)
        return (len(self.data), len(self.data[0]) if self.data else 0)

    def in_bounds(self, r: int, c: int) -> bool:
        # Check if a cell index is inside grid bounds
        # 检查单元格索引是否在网格边界内
        rows, cols = self.shape
        return 0 <= r < rows and 0 <= c < cols

    def __getitem__(self, key: tuple) -> int:
        # Safe indexed access with bounds checking
        # 带边界检查的安全索引访问
        r, c = key
        if not self.in_bounds(r, c):
            rows, cols = self.shape
            raise IndexError(f"Grid2D index out of range: ({r},{c}) not in [0..{rows-1}]x[0..{cols-1}]")
        return self.data[r][c]

    def __setitem__(self, key: tuple, value: int) -> None:
        # Safe indexed assignment with bounds checking
        # 带边界检查的安全索引赋值
        r, c = key
        if not self.in_bounds(r, c):
            rows, cols = self.shape
            raise IndexError(f"Grid2D index out of range: ({r},{c}) not in [0..{rows-1}]x[0..{cols-1}]")
        self.data[r][c] = value

    @classmethod
    def from_size(cls, rows: int, cols: int, fill: int = 0) -> "Grid2D":
        # Construct a grid of given size prefilled with a value
        # 按给定尺寸构造并用指定值填充的网格
        return cls([[fill] * cols for _ in range(rows)])

    def __repr__(self) -> str:
        # Debug-friendly textual representation
        # 便于调试的文本表示
        r, c = self.shape
        return f"Grid2D(shape={r}x{c})"

@dataclass
class ListNode:
    # Singly-linked list node that carries a grid and metadata
    # 携带网格与元数据的单向链表节点
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    grid: Grid2D
    bridge_num: int = 0
    graph_num: int = 0
    next: Optional["ListNode"] = None
    array_data: Optional[List[List[int]]] = None #可能的hinter

    def __post_init__(self) -> None:
        # 若提供了 array_data，则以其初始化 grid；否则根据 grid.shape 初始化 m*n 值为 0 的数组
        if self.array_data is not None:
            self.grid = Grid2D(self.array_data)
        else:
            rows, cols = self.grid.shape
            if rows == 0 or cols == 0:
                self.array_data = []
            else:
                self.array_data = [[0 for _ in range(cols)] for _ in range(rows)]

    def get_min_x(self) -> int:
        return self.min_x
    def get_max_x(self) -> int:
        return self.max_x
    def get_min_y(self) -> int:
        return self.min_y
    def get_max_y(self) -> int:
        return self.max_y
    def get_bridge_num(self) -> int:
        return self.bridge_num
    def get_graph_num(self) -> int:
        return self.graph_num
    def get_grid(self) -> Grid2D:
        return self.grid
    def get_array_data(self) -> List[List[int]]:
        # 返回 array_data 的深拷贝
        return [row[:] for row in self.array_data] if self.array_data else []
    def set_min_x(self, v: int) -> None:
        self.min_x = v
    def set_max_x(self, v: int) -> None:
        self.max_x = v
    def set_min_y(self, v: int) -> None:
        self.min_y = v
    def set_max_y(self, v: int) -> None:
        self.max_y = v
    def set_bridge_num(self, v: int) -> None:
        self.bridge_num = v
    def set_graph_num(self, v: int) -> None:
        self.graph_num = v
    def set_grid_data(self, data: List[List[int]]) -> None:
        # 更新 grid 与 array_data 保持一致
        self.grid = Grid2D(data)
        self.array_data = [row[:] for row in data]

    def __repr__(self) -> str:
        r, c = self.grid.shape
        return (f"ListNode(min_x={self.min_x}, max_x={self.max_x}, "
                f"min_y={self.min_y}, max_y={self.max_y}, "
                f"bridge_num={self.bridge_num}, graph_num={self.graph_num}, "
                f"grid_shape={r}x{c})")
class MyList:
    # Simple singly-linked list specialized for ListNode payloads
    # 面向 ListNode 负载的简单单向链表
    def __init__(self, nodes: Optional[Iterable[ListNode]] = None):
        # Initialize empty list and optionally append initial nodes
        # 初始化空链表，并可选地追加初始节点
        self.head: Optional[ListNode] = None
        self.tail: Optional[ListNode] = None
        self._len: int = 0
        if nodes:
            for n in nodes:
                self.append_node(n)

    def append(self, min_x: int, max_x: int, min_y: int, max_y: int,
               grid_data: List[List[int]], bridge_num: int = 0, graph_num: int = 0) -> ListNode:
        # Create a node from raw data and append to tail
        # 由原始数据创建节点并追加到尾部
        node = ListNode(min_x, max_x, min_y, max_y, Grid2D(grid_data), bridge_num, graph_num)
        return self.append_node(node)

    def prepend(self, min_x: int, max_x: int, min_y: int, max_y: int,
                grid_data: List[List[int]], bridge_num: int = 0, graph_num: int = 0) -> ListNode:
        # Create a node and insert at head
        # 创建节点并插入到头部
        node = ListNode(min_x, max_x, min_y, max_y, Grid2D(grid_data), bridge_num, graph_num, next=self.head)
        self.head = node
        if self.tail is None:
            self.tail = node
        self._len += 1
        return node

    def append_node(self, node: ListNode) -> ListNode:
        # Append an existing node to tail, updating tail pointer
        # 将已有节点追加到尾部，并更新尾指针
        node.next = None
        if not self.head:
            self.head = self.tail = node
        else:
            assert self.tail is not None
            self.tail.next = node
            self.tail = node
        self._len += 1
        return node

    def pop_left(self) -> Optional[ListNode]:
        # Pop and return head node; return None if empty
        # 弹出并返回头节点；若为空则返回 None
        if not self.head:
            return None
        node = self.head
        self.head = node.next
        if self.head is None:
            self.tail = None
        node.next = None
        self._len -= 1
        return node

    def __iter__(self) -> Iterator[ListNode]:
        # Forward iterator from head to tail
        # 从头到尾的正向迭代器
        cur = self.head
        while cur:
            yield cur
            cur = cur.next

    def __len__(self) -> int:
        # Return number of nodes in the list
        # 返回链表中的节点数量
        return self._len

    def clear(self) -> None:
        # Reset the list to empty state
        # 重置链表为空状态
        self.head = self.tail = None
        self._len = 0
