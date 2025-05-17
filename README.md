# CSCI 561 - Foundations of Artificial Intelligence
Foundations of symbolic intelligent systems, search, logic, knowledge representation, planning, and learning.

## Assignment 1 
I implemented a Genetic Algorithm to solve a 3D Traveling Salesman Problem (TSP) where the goal was to find the shortest path for a USC student to run errands across campus and return home, visiting each location exactly once. The assignment involved creating an initial population of random paths, selecting parents using a roulette wheel method, performing two-point crossover to generate offspring while ensuring TSP constraints, and optimizing the solution to minimize the total Euclidean distance traveled in a 3D space defined by (x, y, z) coordinates. My program, written in Python, processed input from a file listing city coordinates, computed an optimal tour, and output the total distance and ordered list of locations to a file. This project enhanced my understanding of AI optimization techniques and their practical application to real-world routing problems.

## Assignment 2
For my CSCI-561 Foundations of Artificial Intelligence Homework 2, I developed an AI agent to play Little-Go, a simplified 5x5 version of the Go board game, using techniques like Minimax with alpha-beta pruning and Q-Learning, implemented from scratch in Python. The assignment tasked me with creating an agent named "my_player.py" that reads game states from an input file, processes the board’s previous and current configurations, and outputs moves to an output file, adhering to Go’s Liberty and KO rules while aiming to maximize territory control. This project deepened my skills in game-playing algorithms and reinforcement learning, culminating in a robust agent capable of competing in simulated tournaments.

## Assignment 3
The solution to Homework 3 of CSCI-561, focused on temporal reasoning using POMDPs (Partially Observable Markov Decision Processes). The assignment involves two scenarios:
- The Little Prince — where an agent traverses an environment based on given actions and observations, and the goal is to infer the most probable sequence of hidden states.
- Speech Recognition — where phonemes are mapped to graphemes, and the goal is to recover the most likely grapheme sequence using probabilistic modeling.

The core implementation applies the Viterbi algorithm to infer hidden state sequences using transition, emission (observation), and initial probabilities derived from provided datasets. The solution parses structured input files, normalizes the data into conditional probabilities, and efficiently computes the most likely hidden state sequence based on the observed action-observation pairs. The code handles both task variants using a unified probabilistic framework and meets all performance and format constraints specified in the assignment.
