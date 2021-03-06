"""The db_preprocess module provides funtionality to prepare pgn data for training stages.

Currently the module only provides conversions for pgn data to bitboard files. Those get saved
as sparse scipy arrays.

Typical Usage example:
    # for a directory with many pgn files the command to process a subset of them:
    parse_pgn_match('path/to/pgn', 'regex', 'path/for/white', 'path/for/black')
    # For the process of one single pgn file:
    parse_pgn('path/to/file.pgn', 'path/for/white', 'path/for/black')
"""
import os
import re
import random
import chess
import chess.pgn as pgn
import scipy.sparse
from omicron.util.representation import to_bitboard


def parse_pgn_match(pgn_dir_path, match, out_white, out_black, out_name=None):
    """Parse all pgn files that match the given regular expression.

    Args:
        pgn_dir_path (str): The path to the directory containing the pgn files
        match (str): Regular expression to filter out the files to preproces
        out_white (str): Path to the directory where white win positions
            will get exported to
        out_black (str): Path to the directory where black win positions
            will get exported to
        out_name(str->str, optional): function to map from file name to out name
    """
    files = os.listdir(pgn_dir_path)
    regex = re.compile(match)
    files = [g for g in files if regex.match(g)]
    for game in files:
        if callable(out_name):
            save_name = out_name(game)
        else:
            save_name = out_name
        parse_pgn_to_bitboard(pgn_dir_path+game, out_white, out_black, save_name)


def parse_pgn_to_bitboard(pgn_path, out_white, out_black, out_name=None):
    """Parse a single pgn file to bitboard files.

    This function saves 10 random positions from the game to bitboards and
    saves them as sparse matrizes in scipy format.
    The save name is $out_name-game_number in the file.

    Args:
        pgn_path (str): The Path to the pgn to be converted
        out_white (str): Path to the directory where white win positions
            will get exported to
        out_black (str): Path to the directory where black win positions
            will get exported to
        out_name (str, optional): The name as which the games will be saved
    """
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
            prev_board = chess.Board()
            for hm_number, pos in enumerate(game.mainline()):
                if hm_number <= 19:
                    prev_board = pos.board()
                    continue
                if prev_board.is_capture(pos.move):
                    prev_board = pos.board()
                    continue
                bitboard = to_bitboard(pos.board())
                positions.append(bitboard)
                prev_board = pos.board()
            positions = random.sample(positions, min(10, len(positions)))
            sparse = scipy.sparse.csc_matrix(positions)
            if winner == 'white':
                scipy.sparse.save_npz(out_white+out_name+'-'+str(game_number), sparse)
            if winner == 'black':
                scipy.sparse.save_npz(out_black+out_name+'-'+str(game_number), sparse)


def load_matrix(path):
    """Load the positions from one game and returns them as numpy array.

    Args:
        path (str): Path to the game to be loaded

    Returns:
        numpy.ndarray: The 10 random positions from the game as numpy array
    """
    sparse = scipy.sparse.load_npz(path)
    return sparse.toarray()


def _extract_result(headers):
    """Extract the results from the game and returns the winner as string.

    Args:
        headers (chess.pgn.Headers): Headers from the game in pychess format

    Returns:
        str: A string representing the winner (either white, blac or draw)
    """
    res = headers.get('Result')
    assert res in ['1-0', '0-1', '1/2-1/2'], 'only legal results are allowed'
    if res == '1-0':
        return 'white'
    if res == '0-1':
        return 'black'
    if res == '1/2-1/2':
        return 'draw'
    return res
