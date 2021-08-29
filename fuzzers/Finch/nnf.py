#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import coloredlogs, logging
import sys
import os
from fuzzer import Fuzzer
from settings import config

coloredlogs.install()
logging.basicConfig(level='INFO')
log = logging.getLogger('nnf')

if __name__ == '__main__':
    conf_file = sys.argv[1]
    if not os.path.exists(conf_file):
        log.error('config file is not found')
        sys.exit(1)
    config.read(conf_file)
    f = Fuzzer()
    f.start()
