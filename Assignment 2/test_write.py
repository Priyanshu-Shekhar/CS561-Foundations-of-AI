import os
from write import writeOutput, writeNextInput

def test_writeOutput():
    writeOutput((2, 3))
    with open('output.txt', 'r') as f:
        assert f.read() == "2,3"
    
    writeOutput("PASS")
    with open('output.txt', 'r') as f:
        assert f.read() == "PASS"
    
    print("writeOutput tests passed!")

def test_writeNextInput():
    piece_type = 1
    previous_board = [
        [0, 0, 1, 1, 0],
        [0, 0, 2, 1, 0],
        [0, 0, 2, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    current_board = [
        [0, 0, 1, 1, 0],
        [0, 0, 2, 1, 0],
        [0, 0, 2, 0, 0],
        [0, 2, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    writeNextInput(piece_type, previous_board, current_board)
    
    expected_content = "1\n00110\n00210\n00200\n02000\n00000\n00110\n00210\n00200\n02010\n00000"
    with open('input.txt', 'r') as f:
        assert f.read() == expected_content
    
    print("writeNextInput test passed!")

if __name__ == "__main__":
    test_writeOutput()
    test_writeNextInput()
    print("All tests passed!")