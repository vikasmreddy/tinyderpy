# tinyderpy
*ML for tic tac toe*
tinyderpy uses machine learning to create a tic tac toe player that will never lose. The training data is created using a simple minimax search, and either a linear regression model or a fully connected neural net with 1 hidden layer can be used (both are able to play perfectly).

*If running on Mac, install Docker for Mac here: https://download.docker.com/mac/stable/Docker.dmg*
*Other platforms see: https://docs.docker.com/install/*

*To run docker: (replace /Users/vikasreddy/code/tinyderpy with your local path to the repo)*
docker run -it -p 8888:8888 -v /Users/vikasreddy/code/tinyderpy:/home/tinyderpy tensorflow/tensorflow

*To get into docker:*
docker ps

*then find the container_id of tensorflow/tensorflow and run (replace container_id with the correct id):*
docker exec -it [container_id] bash

*to run a command directly run (replace container_id with the correct id):*
docker exec -it [container_id] bash -c "cd /home/tinyderpy && [one of the commands below]"

*e.g.*
docker exec -it [container_id] bash -c "cd /home/tinyderpy && python train.py save_training_data"

*Run one of these:*
python train.py save_training_data
python train.py use_saved_training_data
python train.py use_saved_model

*How to create training data / train / run:*

*Step 1: Create training data:*
docker exec -it [container_id] bash -c "cd /home/tinyderpy && python train.py save_training_data"

*Step 2: Train using saved training data:*
docker exec -it [container_id] bash -c "cd /home/tinyderpy && python train.py use_saved_training_data"

*Step 3: Execute using saved model:*
docker exec -it [container_id] bash -c "cd /home/tinyderpy && python train.py use_saved_model"
