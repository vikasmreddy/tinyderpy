# tinyderpy
ML for tic tac toe

*To run docker:*
docker run -it -p 8888:8888 -v /Users/vikasreddy/code/tinyderpy:/home/tinyderpy tensorflow/tensorflow

*To get into docker:*
docker ps

then find the container_id of tensorflow/tensorflow and run

docker exec -it 406aaa3fe58a bash -c "cd /home/tinyderpy && [one of the ones below]"
docker exec -it container_id bash

*Run one of these:*

python train.py save_training_data
python train.py use_saved_training_data
python train.py use_saved_model
