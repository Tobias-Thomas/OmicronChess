import chess
import numpy as np


def to_bitboard(board):
    bitboard = np.array([_square_info_to_bitstring(board.piece_at(x)) for x in range(64)])
    bitboard = bitboard.reshape(-1)
    turn = np.array([board.turn])
    castling_rights = np.array(list(bool(board.castling_rights & r) for r in\
                                    [chess.BB_A1,chess.BB_H1,chess.BB_A8,chess.BB_H8]))
    return np.hstack([bitboard, turn, castling_rights])


def _square_info_to_bitstring(info):
    ''' Returns a bitstring which represents the piece on the given square
    '''
    bitstring = np.zeros(12, dtype=np.bool)
    if not info:
        return bitstring

    piece_type = info.piece_type
    piece_color = info.color
    bitstring[_piece_dict[(piece_type, piece_color)]] = 1
    return bitstring


_piece_dict = {
    (1, True): 5, # White Pawn
    (2, True): 4, # White Knight
    (3, True): 3, # White Bishop
    (4, True): 2, # White Rook
    (5, True): 1, # White Queen
    (6, True): 0, # White King
    (1, False): 11, # Black Pawn
    (2, False): 10, # Black Knight
    (3, False): 9, # Black Bishop
    (4, False): 8, # Black Rook
    (5, False): 7, # Black Queen
    (6, False): 6, # Black King
}
