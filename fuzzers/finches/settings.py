import os
import configparser
import coloredlogs, logging

coloredlogs.install()

global log
log = logging.getLogger('nnf')

global config
config = configparser.ConfigParser()

global bar_format
bar_format = '{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}'

class Settings:
    def __init__(self):

        # setup props
        self.props = dict({})
        name = config['PROPS']['Name']
        timeout = int(config['PROPS']['Timeout']) if 'Timeout' in config['PROPS'] else 10
        max_file = int(config['PROPS']['MaxFile']) if 'MaxFile' in config['PROPS'] else 10000
        gpu = int(config['PROPS']['Gpu']) if 'Gpu' in config['PROPS'] else 0
        epoches = int(config['PROPS']['Epoches']) if 'Epoches' in config['PROPS'] else 200
        batch_size = int(config['PROPS']['BatchSize']) if 'BatchSize' in config['PROPS'] else 32
        bitmap = config['PROPS']['Bitmap'] if 'Bitmap' in config['PROPS'] else 'justmiss'

        self.props['name'] = name
        self.props['timeout'] = timeout
        self.props['max_file'] = max_file
        self.props['gpu'] = gpu
        self.props['epoches'] = epoches
        self.props['bitmap'] = bitmap
        self.props['batch_size'] = batch_size

        ## setup dirs
        in_dir = config['DIRS']['InDir']
        out_dir = '{}/programs_1/{}'.format(os.getcwd(), name)

        self.dirs = dict({'in': in_dir, 'out': out_dir })
        self.dirs['topk'] = '{}/topk'.format(out_dir)
        self.dirs['tmp'] = '{}/tmp'.format(out_dir)
        self.dirs['timeout'] = '{}/timeout'.format(out_dir)
        self.dirs['trace'] = '{}/trace'.format(out_dir)
        self.dirs['crash'] = '{}/crash'.format(out_dir)

        ## setup target
        fast = config['TARGETS']['Fast']
        self.targets = dict({'fast': fast})

        ## setup files
        verifier = config['FILES']['Verifier']
        self.files = dict({'verifier': verifier })
        self.files['map'] = '{}/map_file'.format(out_dir)
        self.files['report'] = '{}/report_file'.format(out_dir)
        self.files['plot'] = '{}/plot_file'.format(out_dir)
