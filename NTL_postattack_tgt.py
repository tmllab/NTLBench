import torch
import wandb
from utils.utils import *
from utils.load_utils import *
import pretrain
import attack
import os
import copy

if __name__ == '__main__':
    # Local test
    # os.environ["WANDB_MODE"] = "disabled"
    
    # Load config file from local
    # wandb.init(project='NTLBenchmark', config='config/cifarstl/attack_tgt.yml')
    # wandb.init(project='NTLBenchmark', config='config/cifarstl/attack_tgtun.yml')
    # wandb.init(project='NTLBenchmark', config='config/visda/attack_tgt.yml')
    # wandb.init(project='NTLBenchmark', config='config/visda/attack_tgtun.yml')
    
    # Load config file from sweep
    wandb.init()
    
    config = wandb.config
    setup_seed(config.seed)
    wandbsweep_config_update(config)

    # load data
    if 'patch' in config.domain_src + config.domain_tgt:
        # for ownership verification
        load_data_func = load_data_tntl_patch
    else:
        # for pure datasets
        load_data_func = load_data_tntl
    (dataloader_train, dataloader_val, dataloader_test, 
     datasets_name) = load_data_func(config)
    
    
    # load model
    model_ntl = load_model(config)
    model_ntl.eval()

    # load pretrain
    cprint('load saved parameters', 'magenta')
    # load trained model and evualate on test data
    if config.pretrained_teacher == 'auto':
        save_path = auto_save_name(config)
    else: 
        save_path = config.pretrained_teacher
    cprint(save_path)
    model_ntl.load_state_dict(torch.load(save_path))
    pretrain.trainer_SL.eval_src(config, dataloader_test,
                            model_ntl, datasets_name=datasets_name)
    
    
    # attack
    dataloader_train_srgt = load_surrogate_data(config, dataloader_train)
    # model_surrogate = load_surrogate_model(config)
    # model_surrogate.eval()
    model_surrogate = copy.deepcopy(model_ntl)

    if config.train_surrogate_scratch:
        cprint('train surrogate model from scratch', 'magenta')
        cprint(f'method: {config.how_to_train_surrogate}', 'yellow')

        if 'FT' in config.how_to_train_surrogate:
            # Supervised fine-tuning-based attack methods 
        
            if len(dataloader_train_srgt) > 2:
                raise NotImplementedError
            # adapt to target domain fine-tuning attack
            dataloader_train_srgt = dataloader_train_srgt[::-1]

            if config.how_to_train_surrogate in ('FT_FTAL', 'FT_Direct_ALL'):
                cprint('Fine-Tuning by FTAL(direct_all) on Dt', 'yellow')
                ft_func = attack.trainer_ft.FTAL
            elif config.how_to_train_surrogate == 'FT_RTAL':
                cprint('Fine-Tuning by RTAL on Dt', 'yellow')
                ft_func = attack.trainer_ft.RTAL
            elif config.how_to_train_surrogate == 'FT_Direct_FC':
                cprint('Fine-Tuning by Direct_FC on Dt', 'yellow')
                ft_func = attack.trainer_ft.direct_FC
            elif config.how_to_train_surrogate == 'FT_InitFC_ALL':
                cprint('Fine-Tuning by InitFC_ALL on Dt', 'yellow')
                ft_func = attack.trainer_ft.initFC_all
            elif config.how_to_train_surrogate == 'FT_InitFC_FC':
                cprint('Fine-Tuning by InitFC_FC on Dt', 'yellow')
                ft_func = attack.trainer_ft.initFC_FC
        
        elif 'SFDA' in config.how_to_train_surrogate:
            # unsupervised-based attacks
            if config.how_to_train_surrogate == 'SFDA_SHOT':
                cprint('SFDA_SHOT on Dt', 'yellow')
                ft_func = attack.trainer_sfda_shot.SHOT
            elif config.how_to_train_surrogate == 'SFDA_CoWA':
                cprint('SFDA_CoWA on Dt', 'yellow')
                ft_func = attack.trainer_sfda_cowa.CoWA
            elif config.how_to_train_surrogate == 'SFDA_NRC':
                cprint('SFDA_NRC on Dt', 'yellow')
                ft_func = attack.trainer_sfda_nrc.NRC
            elif config.how_to_train_surrogate == 'SFDA_PLUE':
                cprint('SFDA_PLUE on Dt', 'yellow')
                ft_func = attack.trainer_sfda_plue.PLUE
            elif config.how_to_train_surrogate == 'SFDA_AdaContrast':
                cprint('SFDA_AdaContrast on Dt', 'yellow')
                ft_func = attack.trainer_sfda_adacontrast.ada_contrast
            elif config.how_to_train_surrogate == 'SFDA_DIFO':
                cprint('SFDA_DIFO on Dt', 'yellow')
                ft_func = attack.trainer_sfda_difo.DIFO
        
        # run attack
        ft_func(config, dataloader_train_srgt, dataloader_val, dataloader_test,
                model_surrogate, datasets_name)
        