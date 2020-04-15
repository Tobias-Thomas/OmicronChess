import chess.pgn as pgn
import numpy as np
from omicron.util.representation import to_bitboard
import random
import scipy.sparse


def parse_pgn_to_bitboard(pgn_path, out_white, out_black, out_name=None):
    if not out_name:
        out_name = pgn_path.split('/')[-1].split('.')[0]
    with open(pgn_path) as pgn_file:
        game_number = -1
        while True:
            game_number += 1
            game = pgn.read_game(pgn_file)
            if not game:
                break
            winner = _extract_result(game.headers)
            if winner == 'draw':
                continue
            positions = []
            for hm_number,pos in enumerate(game.mainline()):
                if hm_number <= 19:
                    continue
                bitboard = to_bitboard(pos.board())
                positions.append(bitboard)
            positions = random.sample(positions, 10)
            sparse = scipy.sparse.csc_matrix(positions)
            if winner == 'white':
                scipy.sparse.save_npz(out_white+out_name+'-'+str(game_number), sparse)
            if winner == 'black':
                scipy.sparse.save_npz(out_black+out_name+'-'+str(game_number), sparse)


def load_matrix(path):
    sparse = scipy.sparse.load_npz(path)
    return sparse.toarray()


def _extract_result(headers):
    res = headers.get('Result')
    assert res in ['1-0', '0-1', '1/2-1/2'], 'only legal results are allowed'
    if res == '1-0':
        return 'white'
    if res == '0-1':
        return 'black'
    if res == '1/2-1/2':
        return 'draw'
    return res
