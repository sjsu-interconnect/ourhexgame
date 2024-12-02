from typing import Tuple

BOARD_SIZE: int = 11
# with the pie rule, there is an additional action to swap colors
ACTION_SPACE: int = BOARD_SIZE * BOARD_SIZE + 1

BOARD_SHAPE: Tuple[int, int] = (BOARD_SIZE, BOARD_SIZE)

PLAYER_1_NUMBER: int = 1
PLAYER_2_NUMBER: int = 2


def get_board_coordinate_for_action(action: int) -> Tuple[int, int]:
    return divmod(action, BOARD_SIZE)

def get_action_from_coordinate(row: int, col: int) -> int:
    return row * BOARD_SIZE + col