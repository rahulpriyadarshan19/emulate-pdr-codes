# Emulating 3DPDR

This is the repository for my master's research project, which is an emulator for [3DPDR](https://uclchem.github.io/3dpdr/). It is built using JAX and Diffrax+Equinox+Optax. 

`NeuralODE.py` is a script used to train the emulator, and `visualize.py` is used to generate statistics and visualizations of predictions given by the emulator. 

Here is a cool video showing the training process. The solid lines are log-scaled abundances of chemical species generated by 3DPDR, and the dashed lines are predictions given by the emulator (the epochs are the actual epoch numbers divided by 10). 

<video src="predictions_2/predictions_all.mp4" controls title="Title"></video>