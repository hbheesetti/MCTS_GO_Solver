from board_base import EMPTY, BLACK, WHITE
import random
from board_util import GoBoardUtil
from board import GoBoard
from typing import List, Tuple

from board_base import (
    board_array_size,
    coord_to_point,
    is_black_white,
    is_black_white_empty,
    opponent,
    where1d,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    MAXSIZE,
    NO_POINT,
    PASS,
    GO_COLOR,
    GO_POINT,
)



class SimulationPlayer(object):
    def __init__(self, numSimulations):
        self.numSimulations = numSimulations

    def name(self):
        return "Simulation Player ({0} sim.)".format(self.numSimulations)

    def genmove(self, state, color, rand):
        assert not state.endOfGame()
        rule, moves = self.ruleBasedMoves(state, color, rand)
        numMoves = len(moves)
        score = [0] * numMoves
        for i in range(numMoves):
            move = moves[i]
            score[i] = self.simulate(state, move, rand)
        bestIndex = score.index(max(score))
        best = moves[bestIndex]
        assert best in state.legalMoves()
        return best

    def ruleBasedMoves(self, board: GoBoard, color, rand: bool) -> Tuple[str, List[int]]:
        """
        return: (MoveType, MoveList)
        MoveType: {"Win", "BlockWin", "OpenFour", "Capture", "Random"}
        MoveList: an unsorted List[int], each element is a move
        """
        if rand == False:
            rule, moves = board.detect_n_in_row(color)
            if len(moves) > 0:
                return rule, moves
        result = board.get_empty_points()
        res = result.tolist()
        return "Random", res

    def simulate(self, state, move, rand):
        num_wins = 0
        num_draws = 0
        cur_player = state.current_player
        for _ in range(self.numSimulations):
            board_copy = state.copy()
            board_copy.play_move(move, state.current_player)
            winner = board_copy.detect_five_in_a_row()
            while winner == EMPTY and len(board_copy.get_empty_points()) != 0:
                rule, moves = self.ruleBasedMoves(
                    board_copy, board_copy.current_player, rand)
                # print(self.moveFormatting(moves))
                random_move = random.choice(moves)
                board_copy.play_move(random_move, board_copy.current_player)
                winner = board_copy.detect_five_in_a_row()
            if winner == state.current_player:
                num_wins += 1
            elif winner == EMPTY:
                num_draws += 1
        score= num_wins * (self.numSimulations + 1) + num_draws
        return score
    
    def moveFormatting(self, moves):
        formatted_moves = []
        s = ""
        for i in moves:
            coord = point_to_coord(i, 7)
            move = format_point(coord)
            formatted_moves.append(move)
        formatted_moves.sort()

        for i in formatted_moves:
            s += str(i) + " "
        return s[:-1]

def point_to_coord(point: GO_POINT, boardsize: int) -> Tuple[int, int]:
    """
    Transform point given as board array index 
    to (row, col) coordinate representation.
    Special case: PASS is transformed to (PASS,PASS)
    """
    if point == PASS:
        return (PASS, PASS)
    else:
        NS = boardsize + 1
        return divmod(point, NS)


def format_point(move: Tuple[int, int]) -> str:
    """
    Return move coordinates as a string such as 'A1', or 'PASS'.
    """
    assert MAXSIZE <= 25
    column_letters = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    if move[0] == PASS:
        return "PASS"
    row, col = move
    if not 0 <= row < MAXSIZE or not 0 <= col < MAXSIZE:
        raise ValueError
    return column_letters[col - 1] + str(row)

