```python
from adsl.core import *
import numpy as np


class Book(Asset):
    def __init__(self, scale: P):
        super().__init__(label="Book")
        self.body = self.attach_part(
            "body",
            cube(scale, color=(0.6, 0.3, 0.1), alpha=0.8),
        )


class Books(Asset):
    def __init__(self, width: float, length: float, book_height: float, num_books: int):
        super().__init__(label="Books")
        rng = np.random.default_rng(7)

        def make_book() -> Asset:
            book = Book(scale=(width, length, book_height))
            book = translate_shape(
                book,
                (
                    rng.uniform(-0.05, 0.05),
                    rng.uniform(-0.05, 0.05),
                    0,
                ),
            )
            angle_degrees = rng.uniform(-15.0, 15.0)
            return rotate_shape(book, axis="+z", angle=angle_degrees)

        self.stack = self.attach_part(
            "stack",
            stack_shapes([make_book() for _ in range(num_books)], axis="z"),
        )


class Table(Asset):
    def __init__(self, top_scale: P, leg_scale: P):
        super().__init__(label="Table")

        # Put the feet on z=0 and support the tabletop at the tops of the legs.
        tabletop_center_z = leg_scale[2] + top_scale[2] / 2.0
        tabletop = cube(
            top_scale,
            center=(0.0, 0.0, tabletop_center_z),
            color=(0.4, 0.2, 0.1),
        )
        self.tabletop = self.attach_part("tabletop", tabletop)

        leg_alignments = (
            ("left_front_top", "left_front_bottom"),
            ("right_front_top", "right_front_bottom"),
            ("right_back_top", "right_back_bottom"),
            ("left_back_top", "left_back_bottom"),
        )
        for index, (leg_anchor, tabletop_anchor) in enumerate(leg_alignments, 1):
            leg = cube(leg_scale, color=(0.3, 0.15, 0.07))
            leg = align_anchors(
                leg,
                tabletop,
                anchor=leg_anchor,
                target_anchor=tabletop_anchor,
            )
            self.attach_part(f"leg_{index}", leg)


class TableWithBooks(Asset):
    def __init__(self):
        super().__init__(label="TableWithBooks")
        table = Table(top_scale=(1.0, 0.6, 0.05), leg_scale=(0.08, 0.08, 0.70))
        self.table = self.attach_part("table", table)

        books = Books(width=0.21, length=0.29, book_height=0.05, num_books=3)
        books = align_anchors(
            books,
            table.tabletop,
            anchor="bottom",
            target_anchor="top",
        )
        self.books = self.attach_part("books", books)


scene = TableWithBooks()
```
