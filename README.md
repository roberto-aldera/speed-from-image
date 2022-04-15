# speed-from-image
A simulation environment used to train a network to predict a trajectory given an image of a scene.
Includes generating maze simulations where a robot moves through a particular environment, or "maze".
Robot dynamics are based on proximity to objects, generating a reasonable trajectory as it moves across the course.
These are then used to train a model on how a robot would normally move, so that at inference time we can provide just the image of the environment and receive a trajectory prediction.
