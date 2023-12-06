"""
board.py
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
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


"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See coord_to_point for explanations of the array encoding.
"""
class GoBoard(object):
    def __init__(self, size: int) -> None:
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0
        self.depth = 0
        self.black_capture_history = []
        self.white_capture_history = []
        self.move_history = []

    def add_two_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            self.black_captures += 2
        elif color == WHITE:
            self.white_captures += 2
    
    def get_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            return self.black_captures
        elif color == WHITE:
            return self.white_captures
        
    def calculate_rows_cols_diags(self) -> None:
        if self.size < 5:
            return
        # precalculate all rows, cols, and diags for 5-in-a-row detection
        self.rows = []
        self.cols = []
        for i in range(1, self.size + 1):
            current_row = []
            start = self.row_start(i)
            for pt in range(start, start + self.size):
                current_row.append(pt)
            self.rows.append(current_row)

            start = self.row_start(1) + i - 1
            current_col = []
            for pt in range(start, self.row_start(self.size) + i, self.NS):
                current_col.append(pt)
            self.cols.append(current_col)

        self.diags = []
        # diag towards SE, starting from first row (1,1) moving right to (1,n)
        start = self.row_start(1)
        for i in range(start, start + self.size):
            diag_SE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            if len(diag_SE) >= 4:
                self.diags.append(diag_SE)
        # diag towards SE and NE, starting from (2,1) downwards to (n,1)
        for i in range(start + self.NS, self.row_start(self.size) + 1, self.NS):
            diag_SE = []
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_SE.append(pt)
                pt += self.NS + 1
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_SE) >= 4:
                self.diags.append(diag_SE)
            if len(diag_NE) >= 4:
                self.diags.append(diag_NE)
        # diag towards NE, starting from (n,2) moving right to (n,n)
        start = self.row_start(self.size) + 1
        for i in range(start, start + self.size):
            diag_NE = []
            pt = i
            while self.get_color(pt) == EMPTY:
                diag_NE.append(pt)
                pt += -1 * self.NS + 1
            if len(diag_NE) >= 4:
                self.diags.append(diag_NE)
        assert len(self.rows) == self.size
        assert len(self.cols) == self.size
        assert len(self.diags) == (2 * (self.size - 4) + 1) * 2
    
    def reset(self, size: int) -> None:
        """
        Creates a start state, an empty board with given size.
        """
        self.size: int = size
        self.NS: int = size + 1
        self.WE: int = 1
        self.last_move: GO_POINT = NO_POINT
        self.last2_move: GO_POINT = NO_POINT
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.black_captures = 0
        self.white_captures = 0
        self.depth = 0
        self.black_capture_history = []
        self.white_capture_history = []
        self.move_history = []

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        b.black_captures = self.black_captures
        b.white_captures = self.white_captures
        b.depth = self.depth
        b.black_capture_history = self.black_capture_history.copy()
        b.white_capture_history = self.white_capture_history.copy()
        b.move_history = self.move_history.copy()
        return b

    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        if point == PASS:
            return True
        #board_copy: GoBoard = self.copy()
        #can_play_move = board_copy.play_move(point, color)
        #return can_play_move
        return self.board[point] == EMPTY

    def end_of_game(self) -> bool:
        return self.get_empty_points().size == 0 or (self.last_move == PASS and self.last2_move == PASS)
           
    def get_empty_points(self) -> np.ndarray:
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board_array: np.ndarray) -> None:
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start : start + self.size] = EMPTY

    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Tries to play a move of color on the point.
        Returns whether or not the point was empty.
        """
        if point == -2:
            self.move_history.append(-2)
            self.current_player = opponent(color)
            return True
        if self.board[point] != EMPTY:
            return False
        self.board[point] = color
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        O = opponent(color)
        offsets = [1, -1, self.NS, -self.NS, self.NS+1, -(self.NS+1), self.NS-1, -self.NS+1]
        bcs = []
        wcs = []
        for offset in offsets:
            if self.board[point+offset] == O and self.board[point+(offset*2)] == O and self.board[point+(offset*3)] == color:
                self.board[point+offset] = EMPTY
                self.board[point+(offset*2)] = EMPTY
                if color == BLACK:
                    self.black_captures += 2
                    bcs.append(point+offset)
                    bcs.append(point+(offset*2))
                else:
                    self.white_captures += 2
                    wcs.append(point+offset)
                    wcs.append(point+(offset*2))
        self.depth += 1
        self.black_capture_history.append(bcs)
        self.white_capture_history.append(wcs)
        self.move_history.append(point)
        return True
    
    def undo(self):
        self.board[self.move_history.pop()] = EMPTY
        self.current_player = opponent(self.current_player)
        self.depth -= 1
        bcs = self.black_capture_history.pop()
        for point in bcs:
            self.board[point] = WHITE
            self.black_captures -= 1
        wcs = self.white_capture_history.pop()
        for point in wcs:
            self.board[point] = BLACK
            self.white_captures -= 1
        if len(self.move_history) > 0:
            self.last_move = self.move_history[-1]
        if len(self.move_history) > 1:
            self.last2_move = self.move_history[-2]

    def neighbors_of_color(self, point: GO_POINT, color: GO_COLOR) -> List:
        """ List of neighbors of point of given color """
        nbc: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point: GO_POINT) -> List:
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point: GO_POINT) -> List:
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1,
                point - self.NS + 1,
                point + self.NS - 1,
                point + self.NS + 1]

    def last_board_moves(self) -> List:
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not NO_POINT, not PASS).
        """
        board_moves: List[GO_POINT] = []
        if self.last_move != NO_POINT and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != NO_POINT and self.last2_move != PASS:
            board_moves.append(self.last2_move)
        return board_moves

    def full_board_detect_five_in_a_row(self) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        Checks the entire board.
        """
        for point in range(self.maxpoint):
            c = self.board[point]
            if c != BLACK and c != WHITE:
                continue
            for offset in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                i = 1
                num_found = 1
                while self.board[point + i * offset[0] * self.NS + i * offset[1]] == c:
                    i += 1
                    num_found += 1
                i = -1
                while self.board[point + i * offset[0] * self.NS + i * offset[1]] == c:
                    i -= 1
                    num_found += 1
                if num_found >= 5:
                    return c
        return EMPTY
    
    def detect_five_in_a_row(self) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        Only checks around the last move for efficiency.
        """
        if self.last_move == NO_POINT or self.last_move == PASS:
            return EMPTY
        c = self.board[self.last_move]
        for offset in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            i = 1
            num_found = 1
            while self.board[self.last_move + i * offset[0] * self.NS + i * offset[1]] == c:
                i += 1
                num_found += 1
            i = -1
            while self.board[self.last_move + i * offset[0] * self.NS + i * offset[1]] == c:
                i -= 1
                num_found += 1
            if num_found >= 5:
                return c
        return EMPTY

    def get_result(self):
        """
        Returns: is_terminal, winner
        If the result is a draw, winner = EMPTY
        """
        winner = self.detect_five_in_a_row()
        if winner != EMPTY:
            return True, winner
        elif self.get_captures(BLACK) >= 10:
            return True, BLACK
        elif self.get_captures(WHITE) >= 10:
            return True, WHITE
        elif self.end_of_game():
            return True, EMPTY
        else:
            return False, EMPTY
        
    def staticallyEvaluateForToPlay(self) :
        captures = 0
        opp_captures = 0

        if(self.current_player == BLACK):
            captures = self.black_captures
            opp_captures = self.white_captures
        else:
            captures = self.white_captures
            opp_captures = self.black_captures

        if self.detect_five_in_a_row() == self.current_player or captures == 10:
            score = 100000000000
        elif self.detect_five_in_a_row() == opponent(self.current_player) or opp_captures == 10:
            score = -100000000000
        elif len(self.get_empty_points()) == 0:
            score = 0
        elif self.detect_five_in_a_row() == EMPTY:
            score = self.get_HeuristicScore()
        return score
    
    def get_HeuristicScore(self):
        score = 0
        opp = opponent(self.current_player)
        lines = self.rows + self.cols + self.diags
        for line in lines:
            for i in range(len(line) - 5):
                currentPlayerCount = 0
                opponentCount = 0
                # count the number of stones on each five-line
                for p in line[i:i + 5]:
                    if self.board[p] == self.current_player:
                        currentPlayerCount += 1
                    elif self.board[p] == opp:
                        opponentCount += 1
                # Is blocked
                if currentPlayerCount < 1 or opponentCount < 1:
                    score += 10 ** currentPlayerCount - 10 ** opponentCount
        return score
        
    def detect_n_in_row(self):
        # Checks for a group of n stones in the same direction on the board.
        b5 = []
        w5 = []
        cap_for_b = []
        cap_for_w = []
        _ = []
        bblocks = []
        wblocks = []
        blocks_of_captures = []
        bfour = []
        wfour = []
        lines = self.rows + self.cols + self.diags
        current_color = self.current_player
        for r in lines:
            w5, b5, bfour, wfour, cap_4w, cap_4b, bblocks, wblocks, cap_black, cap_white  = self.has_n_in_list(r, current_color)

        wins = b5
        blocks = w5
        captures = cap_for_b
        for x in cap_for_w:
            if cap_for_w.count(x)*2 + self.white_captures >= 10:
                blocks += [x]

        # prob need to switch out current_colour for smth else
        captureBlocks_black = self.getCaptureBlocks(
                bblocks, lines, current_color)
        
        captureBlocks_white = self.getCaptureBlocks(
                wblocks, lines, current_color)
        
        score = len(wins)*10000000000 - len(blocks)*1000000000 + len(bfour)*10000 - len(wfour)*10000 + len(cap_4b)*10 - len(cap_4w)*10
        # print(score)
        return score + self.staticallyEvaluateForToPlay()

    def getCaptureBlocks(self, opponent_fives, lines, cur_col):
        capturable = []
        for win in opponent_fives:
            for i in win:
                #print("move", self.moveFormatting([i]))
                stone_lines = []  # all the lines the stone is a part of
                for j in lines:
                    for k in j:
                        if i == k:
                            stone_lines.append(j)
                cap = self.identifyIfCapturable(stone_lines, i)
                capturable += cap
        return capturable

    def identifyIfCapturable(self, lines, stone):
        moves = []
        y = len(lines)
        newLines = []
        for i in lines:
            newLines.append(list(reversed(i)))
        newLines += lines
        for line in newLines:
            index = line.index(stone)
            #print(self.moveFormatting(line))
            if index > 0 and index+2 < len(line):
                i = line[index]
                j = line[index+1]
                k = line[index+2]
                l = line[index-1]
               #print(i, j, k, l)
               #print(self.get_color(i), self.get_color(j),
                     # self.get_color(k), self.get_color(l))

                if self.get_color(j) == self.get_color(i):
                    #print(1)
                    if self.get_color(l) == opponent(self.get_color(i)):
                        #print(2)
                        if self.get_color(k) == 0:
                            moves.append(k)
        return moves

    def has_n_in_list(self, list, current_color) -> GO_COLOR:
        """
        Checks if there are n stones in a row.
        Returns BLACK or WHITE if any n in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = self.get_color(list[0])
        counter = 1
        gap_spot = -1
        before_gap_counter = 0
        b5 = []
        w5 = []
        wfour = []
        bfour = []
        cap_4b = []
        cap_4w = []
        # list of stones captured by white vvvv
        cap_white = []
        # list of stones captured by black vvvv
        cap_black = []
        bblock = []
        wblock = []
        for i in range(1, len(list)):
            color = self.get_color(list[i])
            if color == prev:
                # Matching stone
                counter += 1
            elif (gap_spot == -1 and color == EMPTY):
                # there is a potential gap
                gap_spot = i
                before_gap_counter = counter  # store the number of stones before the gap
            else:
                # if there is a second gap ignore the first gap, set empty to the second gap, and subtract the number of stones before the first gap from the counter.
                # this is so that we can keep the stones after the first gap but before the second gap in the count
                if (color == EMPTY):
                    gap_spot = i
                    counter = counter - before_gap_counter
                    before_gap_counter = counter
                # there is a colour change, reset all vars
                else:
                    before_gap_counter = 0
                    counter = 1
                    gap_spot = -1
                    prev = color
            # if at the end of the board or there has been a colour change get the empty spaces
            if (prev != EMPTY and prev != BORDER and (i+1 >= len(list) or self.get_color(list[i+1]) != color)):
                # print("at end of board?", i, counter)
                if (counter >= 4):
                    if(color == BLACK):
                        b5, bblock = self.five_space(b5,gap_spot,list,i, bblock, counter)
                    elif(color == WHITE):
                        w5, wblock = self.five_space(w5,gap_spot,list,i,wblock,counter)
                    
                    # cap_block = self.capture_block(gap_spot,four_colour,list,i)
                # only get fours if there are no fives and the color is correct
                elif (counter == 3 and color == current_color):
                    if(color == BLACK):
                        bfour = self.four_space(bfour, gap_spot, list, i)
                    elif(color == WHITE):
                        wfour = self.four_space(wfour,gap_spot,list,i)
                elif (counter == 2 and self.get_color(list[i-1]) != 0 and i+1 < len(list)):
                    # print("i-3", self.get_color(list[i-3]))
                    # print("i-2", self.get_color(list[i-2]))
                    # print("i-1", self.get_color(list[i-1]))
                    # print("i", self.get_color(list[i]))
                    # print("i+1", self.get_color(list[i]))
                    # There is a possible capture
                    if self.get_color(list[i-3])*self.get_color(list[i-2]) == 2 and color == 0 and i >= 3:
                        '''
                        Check if the pattern is opp,opp,opp,empty
                        '''
                        # The current stone is an empty spot and two stones back is an opponent of the two in a row
                        if self.get_color(list[i-3]) == 2:
                            # The lone opponent stone is whtie
                            cap_4b += [list[i]]
                            cap_black += [([list[i-2], list[i-1]],list[i])]
                        else:
                            # The lone opponent stone is black
                            cap_4w += [list[i]]
                            cap_white += [([list[i-2], list[i-1]],list[i])]
                    elif self.get_color(list[i-3]) == 0 and self.get_color(list[i-1])*color == 2 and i >= 3:
                        '''
                        Check if the pattern is empty,opp,opp,opp
                        '''
                        # The current stone is an opponent of the 2 stones in a row and 3 stones back is an empty spot
                        if color == 2:
                            cap_4b += [list[i-3]]
                            cap_black += [list[i-2], list[i-1]]
                        else:
                            cap_4w += [list[i-3]]
                            cap_white += [list[i-2], list[i-1]]

                    elif self.get_color(list[i-2]) == 0 and self.get_color(list[i+1])*color == 2 and i >= 2:

                        # The current stone is an opponent of the 2 stones in a row and 3 stones back is an empty spot
                        if color == 2:
                            cap_4b += [list[i-2]]
                            cap_black += [([list[i-1], list[i]],list[i-2])]
                        else:
                            cap_4w += [list[i-2]]
                            cap_white += [list[i-1], list[i]]

                    elif self.get_color(list[i+1]) == 0 and self.get_color(list[i-2])*color == 2 and i >= 2:

                        # The current stone is an opponent of the 2 stones in a row and 3 stones back is an empty spot
                        if self.get_color(list[i-1]) == 2:
                            cap_4b += [list[i+1]]
                            cap_black += [([list[i-1], list[i]],list[i+1])]
                        else:
                            cap_4w += [list[i+1]]
                            cap_white += [list[i-1], list[i]]

        # if cap_4w != []:
        #     print("captured by white")
        #     # for col in cap_4w:
        #     #     print("Move", format_point(point_to_coord(col, self.size)))
        #     for s in cap_white:
        #         print(format_point(point_to_coord(s, self.size)))

        # if cap_4b != []:
        #     print("captured by black")
        #     # for col in cap_4b:
        #     #     #print("Move", format_point(point_to_coord(col, self.size)))
        #     for s in cap_black:
        #         print(format_point(point_to_coord(s, self.size)))
        # print(blocks_of_opponent_fives)
        # print("inside n_row", cap_4b, cap_4w)
        # Code for identifying when there is a potential capture win for a player
        # if self.black_captures == 8:
        #     cap_4w = cap_4b+cap_4w
        # if self.white_captures == 8:
        #     cap_4b = cap_4w+cap_4b
        return [w5, b5, bfour, wfour, cap_4w, cap_4b, bblock, wblock, cap_black, cap_white]


    def five_space(self, fivemakingmoves, empty, list, i, block, counter):
            # if there is an empty space append it is the space that completes the block
        if (empty > 0):
            fivemakingmoves.append(list[empty])
            temp = []
            for j in range(0,counter):
                temp.append(list[i-j])
            #temp = [list[i], list[i-1], list[i-2], list[i-3], list[i-4]]
            if (list[empty] in temp):
                temp.remove(list[empty])

            block.append(temp)
            return [fivemakingmoves,block]
            # if there is an empty space before or after the block add them
        btemp = []
        if(i+1 < len(list) and self.board[list[i+1]] == EMPTY):
            btemp.append(list[i+1])
        if (i-4 >= 0 and self.board[list[i-4]] == EMPTY):
            btemp.append(list[i-4])
        if(len(btemp) > 0):
            block.append([list[i], list[i-1], list[i-2], list[i-3]])
        fivemakingmoves += btemp

        return [fivemakingmoves, block]

    def four_space(self, four, empty, list, i):
       # print(four, empty, list, i, 5)
        # if there is an empty space append it is the space that completes the block
        if (empty > 0):
            four.append(list[empty])
            return four
        # if there are at least 2 empty spaces to a side of the block add the first empty space e.g add ..XXX not O.XXX

        if (i+2 < len(list) and self.board[list[i+1]] == EMPTY and self.board[list[i+2]] == EMPTY):
            four.append(list[i+1])
        if (i-3-1 >= 0 and self.board[list[i-3]] == EMPTY and self.board[list[i-3-1]] == EMPTY):
            four.append(list[i-3])
        # for f in four:
        #     print(format_point(point_to_coord(list[f], 5)))
        return four
        
    def get_result_number(self):
        """
        Returns: is_terminal, winner
        If the result is a draw, winner = EMPTY
        """
        winner = self.detect_five_in_a_row()
        if winner == BLACK or self.get_captures(BLACK) >= 10:
            return 10000  
        elif winner == WHITE or self.get_captures(WHITE) >= 10:
            return -10000
        elif self.end_of_game():
            return 0
        else:
            return 0

    def is_terminal(self):
        winner = self.detect_five_in_a_row()
        if winner != EMPTY:
            return True
        elif self.end_of_game():
            return True
        else:
            return False

    def heuristic_eval(self):
        """
        Returns: a very basic heuristic value of the board
        Currently only considers captures
        """
        if self.current_player == BLACK:
            return (self.black_captures - self.white_captures) / 10
        else:
            return (self.white_captures - self.black_captures) / 10

    def state_to_str(self):
        state = np.array2string(self.board, separator='')
        state += str(self.current_player)
        state += str(self.black_captures)
        state += str(self.white_captures)
        return state

if(__name__ == "__main__"):
    board = GoBoard(7)
    score = board.detect_n_in_row()
    print(score)