import os.path
import time
import pandas as pd
import numpy as np

#Parameters

# name of model. Used for saving conventions
name = 'recsys' # 'imusic'

# set sise of data (number of samples). If None (suggested), full datasets are applied.
limit = None

# how often would you like to check results?
show_every_n_batches = 3000

# decide on wether to show full validation statistics. Computational time is high when this is True
full_validation_stats = False

# decide whether to log testing
log_testing = True

# top k products to determine accuracy
top_k = 20

notes = 'Final GRU Model'

#Hyperparameters

# 512 - Number of sequences running through the network in one pass.
batch_size = 512

# 50 - Embedding dimensions
embed_dim = 300

# The dropout drop probability when training on input. If you're network is overfitting, try decreasing this.
x_drop_probability = 0.00

# The dropout keep probability when training on RNN neurons. If you're network is overfitting, try decreasing this.
rnn_keep_probability = 1.00

# 100 - The number of units in the hidden layers.
rnn_size = 200

# 1
num_layers = 1

# Learning rate for training
# typically 0.0001 up to 1: http://datascience.stackexchange.com/questions/410/choosing-a-learning-rate
# best learning_rate = 0.0025
learning_rate = 0.0025

# 10 epochs
num_epochs = 50


# Create model folder for hyperparameters, statistics and the model itself

## Update and get model_counter
model_counter_path = '../../models/model_counter.txt'
# os.path.isfile() method in Python is used to check whether the specified path is an existing regular file or not.
if os.path.isfile(model_counter_path):
    # The open() function returns a file object, which has a read() method for reading the content of the file
    # Read Only (‘r’)
    model_counter_file = open(model_counter_path, 'r') 
    model_count = int(model_counter_file.read())
    model_counter_file.close()
    # Write Only (‘w’)
    model_counter_file = open(model_counter_path, 'w')
    model_counter_file.write(str(model_count + 1))
    model_counter_file.close()
else:
    # Write and Read (‘w+’)
    model_counter_file = open(model_counter_path, 'w+')
    model_count = 1000 # initial model count/number
    model_counter_file.write(str(model_count + 1))
    model_counter_file.close()

## Make model directory
model_path_dir = '../models/model_count/' + str(model_count) + '-' + name + '-' + time.strftime("%y%m%d") + '/'
if not os.path.exists(model_path_dir):
    os.makedirs(model_path_dir)

## Update stats_file
stats_file_path = model_path_dir + name + '-' + time.strftime("%y%m%d%H%M") + '-statsfile' + '.txt'
stats_file = open(stats_file_path, 'w+')
stats_file.write('model number: {}\n'.format(model_count))
stats_file.write('name: {}\n\n'.format(name))
stats_file.write('limit: {}\n'.format(limit))
stats_file.write('batch_size: {}\n'.format(batch_size))
stats_file.write('embed_dim: {}\n'.format(embed_dim))
stats_file.write('x_drop_probability: {}\n'.format(x_drop_probability))
stats_file.write('rnn_keep_probability: {}\n'.format(rnn_keep_probability))
stats_file.write('rnn_size: {}\n'.format(rnn_size))
stats_file.write('num_layers: {}\n'.format(num_layers))
stats_file.write('learning_rate: {}\n'.format(learning_rate))
stats_file.write('num_epochs: {}\n'.format(num_epochs))
stats_file.write('show_every_n_batches: {}\n'.format(show_every_n_batches))
stats_file.write('top_k: {}\n'.format(top_k))
stats_file.write('full_validation_stats: {}\n'.format(full_validation_stats))
stats_file.write('notes: {}\n'.format(notes))
stats_file.close()

# Load Data
def load_our_data(path, limit):
    return pd.read_csv(path, nrows = limit, sep="\t")

if limit == None:
    validation_limit = None
    testing_limit = None
else:
    validation_limit = int(0.2 * limit)
    testing_limit = int(0.2 * limit)

prepared_data_path = "../../data/rsc15/prepared/"

tr_data = load_our_data(path=f"{prepared_data_path}yoochoose-clicks-100k_train_full.txt", limit=limit)
va_data = load_our_data(path=f"{prepared_data_path}yoochoose-clicks-100k_train_valid.txt", limit=validation_limit)
te_data = load_our_data(path=f"{prepared_data_path}yoochoose-clicks-100k_test.txt", limit=testing_limit)

# Data Preprocessing
## Get unique items
# get number of unique products
print('uniques in training  ', np.unique(tr_data['ItemId']).shape[0])
print('uniques in validation', np.unique(va_data['ItemId']).shape[0])
print('uniques in testing   ', np.unique(te_data['ItemId']).shape[0])

# unique item_ids
uniques = np.unique(np.append(np.append(tr_data['ItemId'], va_data['ItemId']), te_data['ItemId']))
depth = uniques.shape[0]
print('\ndepth (unique items) ', depth)
if depth != np.unique(tr_data['ItemId']).shape[0]:
    print('\nWARNING! Number of uniques in training should equal the depth (uniques in full set)')


## Creating a lookup table
def create_lookup_tables(item_ids):    
    
    items_to_int = pd.Series(data=np.arange(len(item_ids)),index=item_ids)
    int_to_items = pd.DataFrame({"ItemId":item_ids,'item_idx':items_to_int[item_ids].values})
    
    return items_to_int, int_to_items

items_to_int, int_to_items = create_lookup_tables(list(uniques))

## Transforming and splitting the data
# 19 - Number of timesteps the rnn should take in
timesteps = 19

session_key='SessionId'
item_key='ItemId'
time_key='Time'

tr_data.sort_values([session_key, time_key], inplace=True)
print(tr_data)


def drop_single_timestep_sessions(data):
    
    ''' Sessions with a single timestep (one event) are dropped 
    as it is not possible to train a model on inputs with no targets '''
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]
    
    return data

def delete_rare_clicked_items(data):
    #delete records of items which appeared less than 5 times
    itemLen = data.groupby('ItemID').size() #groupby itemID and get size of each item
    data = data[np.in1d(train.ItemID, itemLen[itemLen > 4].index)]
    
    return data

tr_data = drop_single_timestep_sessions(tr_data)
