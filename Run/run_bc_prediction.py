import os
import sys

sys.path.insert(0, "/user/work/yl18410/")
sys.path.insert(0, "/user/work/yl18410/new_graphnet")
sys.path.insert(0, "/user/work/yl18410/new_graphnet/graphnet_LPDM_emulator/")
from new_graphnet.graphnet_LPDM_emulator.full_bc_prediction_pipeline import train_bc_prediction_pipeline
from new_graphnet.graphnet_LPDM_emulator.parameter_files.practice_parameters import practice_hparams_list
from new_graphnet.graphnet_LPDM_emulator.parameter_files.train_parameters import hparams_list


if __name__=="__main__":
    #practice = 'Complete'
    #practice = 'Partial'
    practice = 'Complete'
    if practice =='Full' or practice =='Partial':
        print('Using Practice')
        for practice_hparams in practice_hparams_list:
            #inference_only_pipeline(practice_hparams,practice)
            #bc_prediction_pipeline_practice(practice_hparams,practice)
            #bc_prediction_pipeline(practice_hparams,practice)
            train_bc_prediction_pipeline(practice_hparams,practice)
    else:
        print('Using the full dataset')
        for hparams in hparams_list:
            #inference_only_pipeline(hparams,practice)
            #bc_prediction_pipeline_practice(hparams,practice)
            
            #bc_prediction_pipeline(hparams,practice)
            train_bc_prediction_pipeline(hparams,practice)
            