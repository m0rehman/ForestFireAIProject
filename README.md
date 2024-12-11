## Overview
This repository contains code and supporting scripts for predicting the final burned area fraction of a forest fire using a combination of:

### Cellular Automaton (CA) Simulation
A simplified 2D forest-fire model that simulates fire spread in a small grid representing a forest. Each simulation produces a final fraction of burned cells.

### Neural Network Prediction
A feed-forward neural network that uses the initial forest configuration as input and predicts the fraction of the forest that will eventually burn, without running the full CA simulation. This acts as a computational shortcut for scenario testing.

## Running

- Install dependencies using ```pip install -r requirements.txt```
- Train the network ```python src/neural_net.py```
- Test the model ```python src/test_model.py```

#### NOTE: 
If you need to make a venv to get this to run, do so with Python 3.12
