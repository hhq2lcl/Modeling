# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 15:50:45 2018

@author: ly
"""

import pandas as pd 

reject_data = pd.read_csv(r"F:\python\python\Credit\LoanStats_2016Q1\RejectStats_2016Q4\RejectStats_2016Q4.csv", encoding='latin-1',skiprows = 1,low_memory=False)

