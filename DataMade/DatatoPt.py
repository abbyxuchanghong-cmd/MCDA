# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 08:58:27 2024

@author: XCH
"""

import torch
import pandas as pd

sequence = 37

test_data = pd.read_csv(r"D:\2023-Time\01CSV\CTestData.csv")
print("Test data shape: ", test_data.shape)
test = test_data.values.reshape((test_data.shape[0], 2, sequence))

a={'samples':test}
torch.save(a,r"D:\2023-Time\Data+Code\01\UDA\predict_Czech.pt")

data = torch.load(r'D:\2023-Time\Data+Code\01\UDA\predict_Czech.pt')
print(data)