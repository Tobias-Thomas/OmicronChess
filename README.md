# Omicron Chess  
This is a reimplementation, and hopefully afterwards improvement, of the paper "DeepChess: End-to-End Deep Neural Network for Automatic Learning in Chess"  
Their playing engine consists of two parts:
  1. A Deep Belief Network to convert positions to vectors with a minimum amount of Information loss
  2. Another Deep Network building on top of the first one. It's inputs are two positions (which first get converted to vectors with the first net) and then afterwards classified which one is better for white. This part is done in a supervised scenario with positions from computer games.
For playing games afterwards they used a tweaked version of the Alpha-Beta Search Algorithm  

## State of the project  
Currently I am still working on the reimplementation of the original work. The implementation and training of all networks is done with tensorflow. Currently training is explicitly done on the first GPU. This means you need CUDA or ROCM installed.  
My ideas for further improvement or evaluation are an implementation of the UCI Protocol to play as a BOT on lichess, and maybe try to improve the nets by selfplay after training on computer data.

## Usage of the Project
I will add pretrained models later, but currently you first need to download computer games for training of the two networks. I used the 40/15 games from 2014 onward from the [Computer Chess Rating List](https://www.computerchess.org.uk/ccrl). I recommend downloading the monthly versions and then converting them iteratively.

### Conversion of pgn files
After you've downloaded games for training, you can convert them, by using
```python
from omicron.util.db_preprocess import parse_pgn_match
parse_pgn_match(dir_to_pgn_files, reg_exp_to_filter, save_dir_for_white, save_dir_for_black)
```

### Training the Pos2Vec Model
After converting pgn files to input ready data, we can start by training the first part of the engine
```python
from omicron.training.pos2vec import train_pos2vec_model, save_encoder
model, history = train_pos2vec_model(dir_parsed_white, dir_parsed_black, batch_size)
save_encoder(model, save_name, save_dir)
```
The first two arguments of the trainings function are the save directories from the conversion step
