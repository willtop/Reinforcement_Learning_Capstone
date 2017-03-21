# Reinforcement_Learning_Capstone
This repository is for reinforcement learning algorithm development for ESC472 capstone project


For top-level directory:
Run dqn.py to train and play games

To add a new game
Create a new parameters object in game_params.py
The restart() and reach_terminal_state() function in game_wrapper.py might have to be changed
processInput() in BuildingBlocks.py will need to be changed to calculate the reward from the saved states