# board.py

from pieces import Piece

class Board:
    def __init__(self):
        self.squares = [Piece.EMPTY.value for _ in range(64)]
        self.white_to_move = True

    def set_position_from_fen(self, fen):
        # Split the FEN string into parts
        fen_parts = fen.strip().split()
        if len(fen_parts) < 4:
            raise ValueError("Invalid FEN string")

        piece_placement, side_to_move, castling, en_passant = fen_parts[:4]
        ranks = piece_placement.split('/')

        if len(ranks) != 8:
            raise ValueError("Invalid FEN string")

        self.squares = []
        for rank in ranks:
            file = 0
            for char in rank:
                if char.isdigit():
                    empty_squares = int(char)
                    self.squares.extend([Piece.EMPTY.value] * empty_squares)
                    file += empty_squares
                else:
                    piece = self.char_to_piece(char)
                    self.squares.append(piece.value)
                    file += 1

            if file != 8:
                raise ValueError("Invalid FEN string")

        # Side to move
        self.white_to_move = (side_to_move == 'w')

    def char_to_piece(self, char):
        piece_dict = {
            'P': Piece.W_PAWN,
            'N': Piece.W_KNIGHT,
            'B': Piece.W_BISHOP,
            'R': Piece.W_ROOK,
            'Q': Piece.W_QUEEN,
            'K': Piece.W_KING,
            'p': Piece.B_PAWN,
            'n': Piece.B_KNIGHT,
            'b': Piece.B_BISHOP,
            'r': Piece.B_ROOK,
            'q': Piece.B_QUEEN,
            'k': Piece.B_KING,
        }
        return piece_dict.get(char, Piece.EMPTY)

    def display(self):
        for rank in range(7, -1, -1):
            line = ''
            for file in range(8):
                piece = self.squares[rank * 8 + file]
                line += '{:3}'.format(piece)
            print(line)
        print('White to move' if self.white_to_move else 'Black to move')
