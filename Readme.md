
# 4 ways to increase batch size during training
## Prepare dataset
* Load data from https://www.kaggle.com/chetankv/dogs-cats-images
* Copy `training_set` and `test_set` to `data/dog vs cat/dataset`

## Run experiments
* Run docker 
```
./build-docker.sh
docker run -it --gpus all --ipc=host --rm -v `pwd`:/working_dir experiments:latest
```
* In docker container run `./run_experiments.sh`


| n exp | bs | use cudnn benchmark | use amp | apex amp | checkpointing | average epoch time, sec | val acc | best epoch | GPU memory, Gb |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 0 | 102 | True | False | O0 | 0 | 35.92 | 99.5000 | 11 | 9.21 |
| 1 | 111 | False | False | O0 | 0 | 40.33 | 99.5500 | 12 | 9.26 |
| 2 | 214 | False | True | O0 | 0 | 16.20 | 99.5000 | 13 | 9.27 |
| 3 | 199 | False | False | O1 | 0 | 16.46 | 99.5500 | 7 | 8.66 |
| 4 | 212 | False | False | O2 | 0 | 16.24 | 99.5000 | 13 | 9.10 |
| 5 | 218 | False | False | O3 | 0 | 17.23 | 99.1500 | 14 | 9.25 |
| 6 | 260 | False | True | O0 | 1 | 20.96 | 99.4000 | 10 | 7.03 |
| 7 | 265 | False | True | O0 | 2 | 20.29 | 99.4000 | 8 | 7.16 |
| 8 | 285 | False | True | O0 | 3 | 19.31 | 99.4000 | 11 | 7.68 |
| 9 | 480 | False | True | O0 | optimal | 22.46 | 99.3500 | 12 | 8.87 |
| 10 | 960 | False | True | O0 | optimal | 22.48 | 99.0500 | 13 | 8.87 |