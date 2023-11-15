# UTTT

This refactor will serve as the basis for the UVicAI November 28th 2023 UTTT workshop.

Teams will create a bot that plays UTTT (see UTTT.ai for an example of the rules and interactive (very strong) agent)

The "engine" will provide each agent with the board state such that their markers are "+1" and their opponent is "-1".

For technical reasons, the first 4 moves are played without input from your agent. In the tournament these will be mirrored so each agent plays the same random initialization. Think of this as chess bots having to play from a set of known opennings.

Internal visualization can be performed using .draw_board() and .draw_valid_moves(), but a secondary interface is on the way to display an entire game (the entire set of moves).

As more bots are completed, please consider providing documentation and well commented code to our "bot repo" folder for educational/inspirational purposes.

---
Note: this is active code and there will be changes to it
