# encoding=utf8
import numpy as np
from scipy import stats
import pandas as pd

##########################  read data ####################################

train_path  = '../input/msxf_dialog_train.csv'
test_path =   '../input/msxf_dialog_test_2round.csv'

df_test  = pd.read_csv(test_path, sep='\t')

nn_test = np.concatenate(  [np.load('../blending/round2_nn_blending_.npz')['test'], \
                            np.load('../blending/round2_nn_blending_2_.npz')['test']], 1)

df_test['label'] = pd.DataFrame(nn_test).apply(lambda x: stats.mode(x)[0][0], axis=1)
df_test[['conv_index', 'question_id', 'label']].to_csv('../output/blending_vote.csv', sep='\t', index=None)