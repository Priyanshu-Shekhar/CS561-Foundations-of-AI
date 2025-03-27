#!/usr/bin/env python
# -*- coding:utf-8 -*-
# ProjectName: HW2
# FileName: write
# Description: Functions for writing output and input files for the Little-Go game

def writeOutput(result, path="output.txt"):
    """
    Write the result of a move to the output file.
    
    :param result: Either "PASS" or a tuple of (row, col) coordinates
    :param path: Path to the output file (default is "output.txt")
    """
    if result == "PASS":
        res = "PASS"
    else:
        res = f"{result[0]},{result[1]}"
    
    with open(path, 'w') as f:
        f.write(res)

def writeNextInput(piece_type, previous_board, board, path="input.txt"):
    """
    Write the next input state to the input file.
    
    :param piece_type: 1 for black, 2 for white
    :param previous_board: 5x5 list representing the previous board state
    :param board: 5x5 list representing the current board state
    :param path: Path to the input file (default is "input.txt")
    """
    res = f"{piece_type}\n"
    
    for state in (previous_board, board):
        for row in state:
            res += "".join(map(str, row)) + "\n"
    
    with open(path, 'w') as f:
        f.write(res.strip())

# The writePass function is removed as it's redundant with writeOutput