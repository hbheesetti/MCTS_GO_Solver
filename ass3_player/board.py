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
        self.ko_recapture: GO_POINT = NO_POINT
        self.last_move: GO_POINT = NO_POINT
        self.last2_move: GO_POINT = NO_POINT
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(
            self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.calculate_rows_cols_diags()
        self.black_captures = 0
        self.white_captures = 0

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.ko_recapture = self.ko_recapture
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        return b

    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def _is_legal_check_simple_cases(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check the simple cases of illegal moves.
        Some "really bad" arguments will just trigger an assertion.
        If this function returns False: move is definitely illegal
        If this function returns True: still need to check more
        complicated cases such as suicide.
        """
        assert is_black_white(color)
        if point == PASS:
            return True
        # Could just return False for out-of-bounds,
        # but it is better to know if this is called with an illegal point
        assert self.pt(1, 1) <= point <= self.pt(self.size, self.size)
        assert is_black_white_empty(self.board[point])
        if self.board[point] != EMPTY:
            return False
        if point == self.ko_recapture:
            return False
        return True

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        if point == PASS:
            return True
        board_copy: GoBoard = self.copy()
        can_play_move = board_copy.play_move(point, color)
        return can_play_move

    def end_of_game(self) -> bool:
        return self.last_move == PASS \
            and self.last2_move == PASS

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
            board_array[start: start + self.size] = EMPTY

    def is_eye(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check if point is a simple eye for color
        """
        if not self._is_surrounded(point, color):
            return False
        # Eye-like shape. Check diagonals to detect false eye
        opp_color = opponent(color)
        false_count = 0
        at_edge = 0
        for d in self._diag_neighbors(point):
            if self.board[d] == BORDER:
                at_edge = 1
            elif self.board[d] == opp_color:
                false_count += 1
        return false_count <= 1 - at_edge  # 0 at edge, 1 in center

    def _is_surrounded(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        check whether empty point is surrounded by stones of color
        (or BORDER) neighbors
        """
        for nb in self._neighbors(point):
            nb_color = self.board[nb]
            if nb_color != BORDER and nb_color != color:
                return False
        return True

    def _has_liberty(self, block: np.ndarray) -> bool:
        """
        Check if the given block has any liberty.
        block is a numpy boolean array
        """
        for stone in where1d(block):
            empty_nbs = self.neighbors_of_color(stone, EMPTY)
            if empty_nbs:
                return True
        return False

    def _block_of(self, stone: GO_POINT) -> np.ndarray:
        """
        Find the block of given stone
        Returns a board of boolean markers which are set for
        all the points in the block 
        """
        color: GO_COLOR = self.get_color(stone)
        assert is_black_white(color)
        return self.connected_component(stone)

    def connected_component(self, point: GO_POINT) -> np.ndarray:
        """
        Find the connected component of the given point.
        """
        marker = np.full(self.maxpoint, False, dtype=np.bool_)
        pointstack = [point]
        color: GO_COLOR = self.get_color(point)
        assert is_black_white_empty(color)
        marker[point] = True
        while pointstack:
            p = pointstack.pop()
            neighbors = self.neighbors_of_color(p, color)
            for nb in neighbors:
                if not marker[nb]:
                    marker[nb] = True
                    pointstack.append(nb)
        return marker

    def _detect_and_process_capture(self, nb_point: GO_POINT) -> GO_POINT:
        """
        Check whether opponent block on nb_point is captured.
        If yes, remove the stones.
        Returns the stone if only a single stone was captured,
        and returns NO_POINT otherwise.
        This result is used in play_move to check for possible ko
        """
        single_capture: GO_POINT = NO_POINT
        opp_block = self._block_of(nb_point)
        if not self._has_liberty(opp_block):
            captures = list(where1d(opp_block))
            self.board[captures] = EMPTY
            if len(captures) == 1:
                single_capture = nb_point
        return single_capture

    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Tries to play a move of color on the point.
        Returns whether or not the point was empty.
        """

        if self.board[point] != EMPTY:
            return False
        self.board[point] = color
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        O = opponent(color)
        offsets = [1, -1, self.NS, -self.NS, self.NS +
                   1, -(self.NS+1), self.NS-1, -self.NS+1]
        for offset in offsets:
            if self.board[point+offset] == O and self.board[point+(offset*2)] == O and self.board[point+(offset*3)] == color:
                self.board[point+offset] = EMPTY
                self.board[point+(offset*2)] = EMPTY
                if color == BLACK:
                    self.black_captures += 2
                else:
                    self.white_captures += 2
        return True

    def endOfGame(self) -> bool:
        if self.get_empty_points().size == 0 or GO_COLOR(self.detect_five_in_a_row()) != EMPTY or self.black_captures >= 10 or self.white_captures >= 10:
            return True
        return False

    def legalMoves(self):
        moves = self.get_empty_points()
        return moves

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

    def detect_five_in_a_row(self) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        """
        for r in self.rows:
            result = self.has_five_in_list(r)
            if result != EMPTY:
                return result
        for c in self.cols:
            result = self.has_five_in_list(c)
            if result != EMPTY:
                return result
        for d in self.diags:
            result = self.has_five_in_list(d)
            if result != EMPTY:
                return result
        return EMPTY

    def has_five_in_list(self, list) -> GO_COLOR:
        """
        Returns BLACK or WHITE if any five in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = BORDER
        counter = 1
        for stone in list:
            if self.get_color(stone) == prev:
                counter += 1
            else:
                counter = 1
                prev = self.get_color(stone)
            if counter == 5 and prev != EMPTY:
                return prev
        return EMPTY

    def detect_n_in_row(self, current_color):
        # Checks for a group of n stones in the same direction on the board.
        b5 = []
        w5 = []
        four = []
        cap_for_b = []
        cap_for_w = []
        _ = []
        blocks_of_opponent_fives = []
        blocks_of_captures = []
        lines = self.rows + self.cols + self.diags
        for r in lines:
            rows = self.has_n_in_list(r, current_color)
            w5 += rows[0]
            b5 += rows[1]
            four += rows[2]
            cap_for_w += rows[3]
            cap_for_b += rows[4]
            blocks_of_opponent_fives += rows[5]
            blocks_of_captures += rows[6]
        if current_color == BLACK:
            wins = b5
            blocks = w5
            captures = cap_for_b
            for x in cap_for_w:
                if cap_for_w.count(x)*2 + self.white_captures >= 10:
                    blocks += [x]
        elif current_color == WHITE:
            wins = w5
            blocks = b5
            captures = cap_for_w
            for x in cap_for_b:
                if cap_for_b.count(x)*2 + self.black_captures >= 10:
                    blocks += [x]
        if (len(wins) > 0):
            return "Win", wins
        elif len(blocks) > 0:
            captureBlocks = self.getCaptureBlocks(
                blocks_of_opponent_fives, lines, current_color)
            return "BlockWin", blocks+captureBlocks
        elif len(four) > 0:
            return "OpenFour", four
        elif len(captures) > 0:
            return "Capture", captures
        return "none", []

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

    def moveFormatting(self, moves):
        formatted_moves = []
        s = ""
        for i in moves:
            coord = point_to_coord(i, self.size)
            move = format_point(coord)
            formatted_moves.append(move)
        formatted_moves.sort()

        for i in formatted_moves:
            s += str(i) + " "
        return s[:-1]

    def has_n_in_list(self, list, current_color) -> GO_COLOR:
        """
        Checks if there are n stones in a row.
        Returns BLACK or WHITE if any n in a rows exist in the list.
        EMPTY otherwise.
        """
        prev = self.get_color(list[0])
        counter = 1
        gap_spot = 0
        before_gap_counter = 0
        b5 = []
        w5 = []
        four = []
        cap_4b = []
        cap_4w = []
        # list of stones captured by white vvvv
        cap_white = []
        # list of stones captured by black vvvv
        cap_black = []
        blocks_of_opponent_fives = []
        for i in range(1, len(list)):
            color = self.get_color(list[i])
            if color == prev:
                # Matching stone
                counter += 1
            elif (gap_spot == 0 and color == EMPTY):
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
                    gap_spot = 0
                    prev = color
            # if at the end of the board or there has been a colour change get the empty spaces
            if (prev != EMPTY and prev != BORDER and (i+1 >= len(list) or self.get_color(list[i+1]) != color)):
                # print("at end of board?", i, counter)
                if (counter >= 4):
                    w5, b5, blocks_of_opponent_fives = self.five_space(
                        w5, b5, gap_spot, list, i, color, blocks_of_opponent_fives, current_color)
                    
                    # cap_block = self.capture_block(gap_spot,four_colour,list,i)
                # only get fours if there are no fives and the color is correct
                elif (counter == 3 and color == current_color):
                    four = self.four_space(four, gap_spot, list, i)
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
        if current_color == 2:
            return [w5, b5, four, cap_4w, cap_4b, blocks_of_opponent_fives, cap_black]
        else:
            return [w5, b5, four, cap_4w, cap_4b, blocks_of_opponent_fives, cap_white]

    def five_space(self, w, b, empty, list, i, color, block, current_color):
        if (color == BLACK):
            # if there is an empty space append it is the space that completes the block
            if (empty > 0):
                b.append(list[empty])
                temp = [list[i], list[i-1], list[i-2], list[i-3], list[i-4]]
                temp.remove(list[empty])
                #temp.append(list[i-4])
                block.append(temp)
                return [w,b,block]
            # if there is an empty space before or after the block add them 
            if(i+1 < len(list) and self.board[list[i+1]] == EMPTY):
                b.append(list[i+1])
            if (i-4 >= 0 and self.board[list[i-4]] == EMPTY):
                b.append(list[i-4])
            if(len(b) > 0 and current_color != BLACK):
                block.append([list[i], list[i-1], list[i-2], list[i-3]])
            return [w,b,block]
            
        elif(color == WHITE):
            if(empty > 0):
                # if there is an empty space append it is the space that completes the block
                temp = [list[i], list[i-1], list[i-2], list[i-3], list[i-4]]
                temp.remove(list[empty])
                #temp.append(list[i-4])
                block.append(temp)
                return [w,b,block]
            # if there is an empty space before or after the block add them 
            if(i+1 < len(list) and self.board[list[i+1]] == EMPTY):
                w.append(list[i+1])
            if(i-4 >= 0 and self.board[list[i-4]] == EMPTY):
                w.append(list[i-4]) 
            if(len(w) > 0 and current_color != WHITE):
                block.append([list[i], list[i-1], list[i-2], list[i-3]])
            return [w,b,block]

        return [w, b, block]

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

    def capture_block(self, gap, colour, list, i):
        """start = list[i-4] # get start of the block
        end = list[i]

        if(self.board[end] != opponent(colour)):
            return

        # 
        if(end-8 < 0 or end-8 >):


        top_list_opp = []
        top_list_col = []
        for n in range(7):
            check_colour = self.board[start+n-8]
            if(start+n == gap):
                continue
            elif(check_colour == opponent(colour)):
                top_list_opp += start+n-8
            elif(check_colour == colour):
                top_list_col += start+n+8

        for space in top_list """


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
