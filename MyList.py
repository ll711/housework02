"""
Hinger Project
Coursework 001 for: CMP-6058A Artificial Intelligence

Includes a State class for Task 1

@author: B20 (100528137 and 100531086)
@date: 11/10/2025

"""
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Iterator

from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Iterator

__all__ = ["Grid2D", "ListNode", "MyList"]

class Grid2D:
    # Lightweight 2D grid wrapper with bounds checking and shape helpers
    def __init__(self, data: List[List[int]]):
        # Enforce equal-length rows to keep a proper rectangle grid
        if data and any(len(row) != len(data[0]) for row in data):
            raise ValueError("Grid2D rows must have equal length")
        self.data = data or []

    @property
    def shape(self) -> Tuple[int, int]:
        # Return grid dimensions as (rows, cols)
        return (len(self.data), len(self.data[0]) if self.data else 0)

    def in_bounds(self, r: int, c: int) -> bool:
        # Check if a cell index is inside grid bounds
        rows, cols = self.shape
        return 0 <= r < rows and 0 <= c < cols

    def __getitem__(self, key: tuple) -> int:
        # Safe indexed access with bounds checking
        r, c = key
        if not self.in_bounds(r, c):
            rows, cols = self.shape
            raise IndexError(f"Grid2D index out of range: ({r},{c}) not in [0..{rows-1}]x[0..{cols-1}]")
        return self.data[r][c]

    def __setitem__(self, key: tuple, value: int) -> None:
        # Safe indexed assignment with bounds checking
        r, c = key
        if not self.in_bounds(r, c):
            rows, cols = self.shape
            raise IndexError(f"Grid2D index out of range: ({r},{c}) not in [0..{rows-1}]x[0..{cols-1}]")
        self.data[r][c] = value

    @classmethod
    def from_size(cls, rows: int, cols: int, fill: int = 0) -> "Grid2D":
        # Construct a grid of given size prefilled with a value
        return cls([[fill] * cols for _ in range(rows)])

    def __repr__(self) -> str:
        # Debug-friendly textual representation
        r, c = self.shape
        return f"Grid2D(shape={r}x{c})"

@dataclass
class ListNode:
    # Singly-linked list node that carries a grid and metadata
    min_x: int
    max_x: int
    min_y: int
    max_y: int
    grid: Grid2D
    bridge_num: int = 0
    graph_num: int = 0
    next: Optional["ListNode"] = None
    array_data: Optional[List[List[int]]] = None  # possible hinger markers

    def __post_init__(self) -> None:
        # If array_data is provided, initialize grid from it; otherwise initialize array_data
        # as an m x n zero array based on grid.shape
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

    def set_array_data(self, data: List[List[int]]) -> None:
        """Update the node's array_data (deep copy to avoid side effects)"""
        self.array_data = [row[:] for row in data]  # deep copy

    def get_array_data(self) -> List[List[int]]:
        # Return a deep copy of array_data
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
        # Update grid and keep array_data consistent
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
    def __init__(self, nodes: Optional[Iterable[ListNode]] = None):
        # Initialize empty list and optionally append initial nodes
        self.head: Optional[ListNode] = None
        self.tail: Optional[ListNode] = None
        self._len: int = 0
        if nodes:
            for n in nodes:
                self.append_node(n)

    def append(self, min_x: int, max_x: int, min_y: int, max_y: int,
               grid_data: List[List[int]], bridge_num: int = 0, graph_num: int = 0) -> ListNode:
        # Create a node from raw data and append to tail
        node = ListNode(min_x, max_x, min_y, max_y, Grid2D(grid_data), bridge_num, graph_num)
        return self.append_node(node)

    def prepend(self, min_x: int, max_x: int, min_y: int, max_y: int,
                grid_data: List[List[int]], bridge_num: int = 0, graph_num: int = 0) -> ListNode:
        # Create a node and insert at head
        node = ListNode(min_x, max_x, min_y, max_y, Grid2D(grid_data), bridge_num, graph_num, next=self.head)
        self.head = node
        if self.tail is None:
            self.tail = node
        self._len += 1
        return node

    def append_node(self, node: ListNode) -> ListNode:
        # Append an existing node to tail, updating tail pointer
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
        cur = self.head
        while cur:
            yield cur
            cur = cur.next

    def __len__(self) -> int:
        # Return number of nodes in the list
        return self._len

    def clear(self) -> None:
        # Reset the list to empty state
        self.head = self.tail = None
        self._len = 0
