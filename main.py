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

def make_batches(ds, buffer_size, batch_size):
    return ds.cache().shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def create_tensor_dataset(encoder,bs, split=True):
    """bs is Batch Size
       if split=True, the input data is split into train and validation, otherwise the whole data is used for training """
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

    raw_data = pd.read_csv('../DATA/tr_by_acct_w_age.csv')
    data, LOG_AMOUNT_SCALE, TD_SCALE,ATTR_SCALE, START_DATE, TCODE_TO_NUM, NUM_TO_TCODE = preprocess_data_czech(raw_data)
    data2 = data[['account_id','age','age_sc', 'tcode', 'tcode_num', 'datetime', 'month', 'dow', 'day','td', 'dtme', 'log_amount','log_amount_sc','td_sc']]
    df= data2.copy()

    n_tcodes = len(TCODE_TO_NUM)

    info = FieldInfo(n_tcodes)

    max_seq_len = 80
    min_seq_len = 20
    batch_size = 64
    encoder = TensorEncoder(df, info, max_seq_len, min_seq_len)

    encoder.encode()

    n_seqs, n_steps, n_feat_inp = encoder.inp_tensor.shape

    train_batches, val_batches = create_tensor_dataset(encoder,batch_size, split=True)

    #x_tr, x_cv, inds_tr, inds_cv, targ_tr, targ_cv = train_test_split(encoder.inp_tensor, np.arange(n_seqs), encoder.tar_tensor, test_size=0.2)

    # # Create TensorFlow dataset
    # ds_all = tf.data.Dataset.from_tensor_slices((encoder.inp_tensor.astype(np.float32), encoder.tar_tensor.astype(np.float32)))
    # ds_tr = tf.data.Dataset.from_tensor_slices((x_tr.astype(np.float32), targ_tr.astype(np.float32)))
    # ds_cv = tf.data.Dataset.from_tensor_slices((x_cv.astype(np.float32), targ_cv.astype(np.float32)))

    # BUFFER_SIZE = ds_all.cardinality().numpy()
    # bs = 64  # batch size


    # train_batches = make_batches(ds_tr, BUFFER_SIZE, bs)
    # val_batches =  make_batches(ds_cv, BUFFER_SIZE, bs)

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

    features = 26
    d_embedding = 128
    dff = 128
    d_model = 128
    batch_size = 64
    seq_len = 80
    maximum_position_encoding = 256
    rate = 0.1
    num_heads = 2
    num_layers = 4
    raw_features = 7
    transformer = Transformer(features,dff, d_embedding, d_model, maximum_position_encoding,num_heads, num_layers,config, rate=0.1)
    epochs = 80
    early_stop = 2
    train = Train(transformer)
    with  tf.device('/gpu:0'):
        train.train(train_batches, val_batches, x_cv, targ_cv, epochs, early_stop)
        attributes = encoder.attributes
        synth = train.generate_synthetic_data(80, 5000, df, attributes, n_feat_inp)
        filename = '../DATA/synth_transformer' + 'v1' +'.csv'
        synth.to_csv(filename, index=False)
        print('finish')

if __name__ == "__main__":
    main()

