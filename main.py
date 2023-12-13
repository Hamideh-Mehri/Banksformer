from prepare_data import preprocess_data_czech
from field_info import FieldInfo
from tensor_encoder import TensorEncoder
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from train import Train
import tensorflow as tf
from modules import Transformer
import time
import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def make_batches(ds, buffer_size, batch_size):
    return ds.cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_tensor_dataset(encoder,bs, split=True):
    """bs is Batch Size
       if split=True, the input data is split into train and validation, otherwise the whole data is used for training """
    n_seqs, _, _ = encoder.inp_tensor.shape

    x_tr, x_cv, inds_tr, inds_cv, targ_tr, targ_cv = train_test_split(encoder.inp_tensor, np.arange(n_seqs), encoder.tar_tensor, test_size=0.2)

    # Create TensorFlow dataset
    ds_all = tf.data.Dataset.from_tensor_slices((encoder.inp_tensor.astype(np.float32), encoder.tar_tensor.astype(np.float32)))
    ds_tr = tf.data.Dataset.from_tensor_slices((x_tr.astype(np.float32), targ_tr.astype(np.float32)))
    ds_cv = tf.data.Dataset.from_tensor_slices((x_cv.astype(np.float32), targ_cv.astype(np.float32)))

    BUFFER_SIZE = ds_all.cardinality().numpy()

    all_batches =   make_batches(ds_all, BUFFER_SIZE, bs)
    train_batches = make_batches(ds_tr, BUFFER_SIZE, bs)
    val_batches =  make_batches(ds_cv, BUFFER_SIZE, bs)

    if split:
        return train_batches, val_batches
    else:
        return all_batches
    

def main():
    
    confighyper = load_config('config_hyper.json')

    max_seq_len = confighyper['max_seq_len']
    min_seq_len = confighyper['min_seq_len']
    batch_size = confighyper['batch_size']
    d_embedding = confighyper['d_embedding']
    dff = confighyper['dff']
    d_model = confighyper['d_model']
    maximum_position_encoding = confighyper['maximum_position_encoding']
    rate = confighyper['rate']
    num_heads = confighyper['num_heads']
    num_layers = confighyper['num_layers']   # number of decoder layers that is stacked in transformer's Decoder
    epochs = confighyper['epochs'] 
    early_stop = confighyper['early_stop'] 
    len_generated_seq = confighyper['len_generated_seq'] 
    num_generated_seq = confighyper['num_generated_seq'] 
    synth_data_filename = confighyper["synth_data_filename"]


    raw_data = pd.read_csv('../DATA/tr_by_acct_w_age.csv')
    data, LOG_AMOUNT_SCALE, TD_SCALE,ATTR_SCALE, START_DATE, TCODE_TO_NUM, NUM_TO_TCODE = preprocess_data_czech(raw_data)
    selected_data_columns = data[['account_id','age','age_sc', 'tcode', 'tcode_num', 'datetime', 'month', 'dow', 'day','td', 'dtme', 'log_amount','log_amount_sc','td_sc']]
    df= selected_data_columns.copy()

    n_tcodes = len(TCODE_TO_NUM)

    info = FieldInfo(n_tcodes)

    
    encoder = TensorEncoder(df, info, max_seq_len, min_seq_len)
    encoder.encode()

    n_seqs, seq_len, n_feat_inp = encoder.inp_tensor.shape
    raw_features = encoder.tar_tensor.shape[-1]    #7

    train_batches, val_batches = create_tensor_dataset(encoder,batch_size, split=True)

    ACTIVATIONS = {
    "td_sc": "relu",
    "log_amount_sc": "relu"
    }
    fieldInfo = FieldInfo(n_tcodes)
    config = {}
    config["ORDER"] = fieldInfo.DATA_KEY_ORDER
    config["FIELD_STARTS_IN"] = fieldInfo.FIELD_STARTS_IN
    config["FIELD_DIMS_IN"] = fieldInfo.FIELD_DIMS_IN
    config["FIELD_STARTS_NET"] = fieldInfo.FIELD_STARTS_NET
    config["FIELD_DIMS_NET"] = fieldInfo.FIELD_DIMS_NET
    config["ACTIVATIONS"] = ACTIVATIONS


    transformer = Transformer(n_feat_inp, dff, d_embedding, d_model, maximum_position_encoding,num_heads, num_layers,config, rate=0.1)
   
    train = Train(transformer)
    with  tf.device('/gpu:0'):
        train.train(train_batches, val_batches, epochs, early_stop)
        attributes = encoder.attributes
        synth = train.generate_synthetic_data(len_generated_seq, num_generated_seq, df, attributes, n_feat_inp)
        
        synth.to_csv(synth_data_filename, index=False)
        print('finish')

if __name__ == "__main__":
    main()

