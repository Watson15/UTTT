# UTTT

This code will serve as the basis for the UVicAI November 28th 2023 UTTT workshop.

Teams will create a bot that plays UTTT. See [UTTT.ai](https://www.uttt.ai/) for rules and an interactive opponent.
The notebook <b>uttt - getting started</b> is the suggested starting point for understanding the game engine and the demands of out bot.

Internal visualization can be performed using .draw_board() and .draw_valid_moves(), but a secondary interface is available [here](https://github.com/NathanPannell/uttt-visual).

As more bots are completed, please consider providing documentation and well commented code to our "bot repo" folder for educational/inspirational purposes.

--- 
The recent refactor made the engine easier to work with alternating agents and the upcoming tournament:
 - The "engine" will provide each agent with the board state such that their markers are "+1" and their opponent is "-1".
 - For technical reasons, the first 4 moves are played without input from your agent. In the tournament these will be mirrored so each agent plays the same random initialization. Think of this as chess bots having to play from a set of known opennings.

---
Note: this is active code and there will be changes to it
