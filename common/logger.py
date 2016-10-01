# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 19:27:22 2016

@author: george
"""

import logging
import sys

"""
	Initialization of system logger. 
"""

logger = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def getLogger():
	return logger

