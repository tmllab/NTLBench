from .ntl_utils.utils_digit import get_mnist_data, get_syn_data, get_mnist_m_data, get_svhn_data, get_usps_data
from .ntl_utils.utils_digit import get_mnist_color_10, get_mnist_color_20, get_mnist_color_90
from .ntl_utils.utils_digit import get_mnist_rotate_0, get_mnist_rotate_15, get_mnist_rotate_30, get_mnist_rotate_45, get_mnist_rotate_60, get_mnist_rotate_75
from .ntl_utils.getdata import Cus_Dataset
from .ntl_utils.utils_cifar_stl import get_cifar_data, get_stl_data
from .ntl_utils.utils_visda import get_visda_data_tgt, get_visda_data_src
from .ntl_utils.utils_officehome import get_home_art, get_home_cli, get_home_pd, get_home_rw
from .ntl_utils.utils_imagenette import get_imagenette_data
from .ntl_utils.utils_vlcs import get_vlcs_V, get_vlcs_C, get_vlcs_L, get_vlcs_S
from .ntl_utils.utils_pacs import get_pacs_P, get_pacs_A, get_pacs_C, get_pacs_S
from .ntl_utils.utils_terra_incognita import get_ti_l38, get_ti_l43, get_ti_l46, get_ti_l100
from .ntl_utils.utils_domain_net import get_domain_net_cli, get_domain_net_info, get_domain_net_paint, get_domain_net_qd, get_domain_net_real, get_domain_net_sketch
import models.ntl_vggnet as ntl_vggnet
import models.ntl_vit as ntl_vit
import torch
import numpy as np
import random 
from termcolor import cprint
import torchvision
import sys


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

domain_dict = {
               'mt': get_mnist_data,
               'us': get_usps_data,
               'sn': get_svhn_data,
               'mm': get_mnist_m_data,
               'sd': get_syn_data,
               'cifar': get_cifar_data,
               'stl': get_stl_data,
               'visda_t': get_visda_data_src,
               'visda_v': get_visda_data_tgt,
               'home_art': get_home_art, 
               'home_cli': get_home_cli, 
               'home_pd': get_home_pd, 
               'home_rw': get_home_rw,
               'vlcs_v': get_vlcs_V,
               'vlcs_c': get_vlcs_C,
               'vlcs_l': get_vlcs_L,
               'vlcs_s': get_vlcs_S,
               'pacs_p': get_pacs_P,
               'pacs_a': get_pacs_A,
               'pacs_c': get_pacs_C,
               'pacs_s': get_pacs_S,
               'ti_l38': get_ti_l38,
               'ti_l43': get_ti_l43,
               'ti_l46': get_ti_l46,
               'ti_l100': get_ti_l100,
               'domain_net_cli': get_domain_net_cli,
               'domain_net_info': get_domain_net_info,
               'domain_net_paint': get_domain_net_paint,
               'domain_net_qd': get_domain_net_qd,
               'domain_net_real': get_domain_net_real,
               'domain_net_sketch': get_domain_net_sketch,
               'cmt10': get_mnist_color_10,
               'cmt20': get_mnist_color_20,
               'cmt90': get_mnist_color_90,
               'rmt0': get_mnist_rotate_0,
               'rmt15': get_mnist_rotate_15,
               'rmt75': get_mnist_rotate_75
               }

model_dict = {'vgg11': ntl_vggnet.vgg11,
              'vgg11bn': ntl_vggnet.vgg11_bn,
              'vgg13': ntl_vggnet.vgg13,
              'vgg13bn': ntl_vggnet.vgg13_bn,
              'vgg19': ntl_vggnet.vgg19,
              'vgg19bn': ntl_vggnet.vgg19_bn,
              'vit_tiny': ntl_vit.vit_tiny,
              'vit_base': ntl_vit.vit_base,
              'vit_large': ntl_vit.vit_large,
              'vit_huge': ntl_vit.vit_huge
              }


model_dict_cmi = {'resnet50': 'resnet50_imagenet',
                  'resnet34': 'resnet34_imagenet',
                  'resnet18': 'resnet18_imagenet',
                  'wide_resnet50_2': 'wide_resnet50_2_imagenet'}


domain_digits = ['mt', 'us', 'sn', 'mm', 'sd']
domain_cifar_stl = ['cifar', 'stl']
domain_visda = ['visda_v', 'visda_t']
domain_vlcs = ['vlcs_v', 'vlcs_c', 'vlcs_l', 'vlcs_s']
domain_officehome = ['home_art', 'home_cli', 'home_pd', 'home_rw']
domain_imagenette = ['imagenette']
domain_pacs = ['pacs_p', 'pacs_a', 'pacs_c', 'pacs_s']
domain_terra_incognita = ['ti_l38', 'ti_l43', 'ti_l46', 'ti_l100']
domain_domain_net = ['domain_net_cli', 'domain_net_info', 'domain_net_paint', 'domain_net_qd', 'domain_net_real', 'domain_net_sketch']

def auto_save_name(config):
    if config.task_name in ['SL', 'sNTL', 'sCUTI']:
        save_path = f'./saved_models/{config.task_name}_{config.domain_src}_{config.teacher_network}.pth'
    else:
        save_path = f'./saved_models/{config.task_name}_{config.domain_src}_{config.domain_tgt}_{config.teacher_network}.pth'
    return save_path

def is_debug_mode():
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        return False
    else:
        return gettrace() is not None

def wandbsweep_config_update(config):
    if 'sweep_domains' in config.keys():
        domain_src, domain_tgt = config.sweep_domains
        config_update = {'domain_src': domain_src,
                         'domain_tgt': domain_tgt}
        config.update(config_update, allow_val_change=True)
        
    # update num_classes
    config.update({'num_classes': 10}, allow_val_change=True) # default: 10
    num_classes_map = {
            'domain_net': 345,
            'home': 65,
            'pacs': 7,
            'vlcs': 5,
            'visda': 12,
            'cmt': 2
        }
    for domain, num_classes in num_classes_map.items():
        if domain in config.domain_src:
            config.update({'num_classes': num_classes}, allow_val_change=True)
            break
    
    # tSOPHON pretrain config
    if config.task_name == 'tSOPHON' and not hasattr(config, 'how_to_train_surrogate'):
        if not hasattr(config, 'SOPHON_alpha_weight'):
            config.update({'SOPHON_alpha_weight': 1.0}, allow_val_change=True)
        sophon_config = {
            'inverse_loss':{
                'SOPHON_nl_loop': 1,
                'SOPHON_alpha': 3.0 * config.SOPHON_alpha_weight,
                'SOPHON_beta': 5.0},
            'kl_loss':{
                'SOPHON_nl_loop': 5,
                'SOPHON_alpha': 1.0 * config.SOPHON_alpha_weight,
                'SOPHON_beta': 1.0},
        }
        spec_sophon_config = sophon_config[config.SOPHON_maml_loss_func]
        config.update(spec_sophon_config, allow_val_change=True)
    
    
    # apply vit_tiny for tSOPHON, avoid out-of-memory
    if config.task_name == 'tSOPHON' and config.teacher_network == 'vit_base':
        config.update({'teacher_network': 'vit_tiny'}, allow_val_change=True)

