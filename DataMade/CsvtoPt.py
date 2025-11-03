# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:05:15 2023

@author: XCH
"""

import torch
import numpy as np
import pandas as pd

sample1=pd.read_csv(r'D:\2023-Time\Data\02\O\042-RiceTest\VI\RTestNDVI.csv')
sample2=pd.read_csv(r'D:\2023-Time\Data\02\O\042-RiceTest\VI\RTestEVI2.csv')
sample3=pd.read_csv(r'D:\2023-Time\Data\02\O\042-RiceTest\VI\RTestLSWI.csv')
sample4=pd.read_csv(r'D:\2023-Time\Data\02\O\042-RiceTest\VI\RTestNDWI.csv')
sample5=pd.read_csv(r'D:\2023-Time\Data\02\O\042-RiceTest\VI\RTestMNDWI.csv')
label=pd.read_csv(r'D:\2023-Time\Data\02\O\042-RiceTest\VI\MTestLabel.csv')
sample=pd.concat([sample1,sample2,sample3,sample4,sample5], axis=1)

sample=np.array(sample)

sample=sample.reshape(5000, 5, 22)
label=np.array(label)
label=label.reshape(5000)
a={'samples':sample,'labels':label}
torch.save(a,r"D:\2023-Time\Data\02\UDA\test_Rice.pt")

data = torch.load(r'D:\2023-Time\Data\02\UDA\test_Rice.pt')
print(data)

"""
sample=pd.read_csv(r'D:\2023-Time\TSCC\DataMade\HLJTest.csv')
sample=np.array(sample)
sample=sample.reshape(5000, 22, 1)
a={'samples':sample}
# b={'samples':sample}
torch.save(a,r"D:\2023-Time\TSCC\Data\KtoC\predict_Czech.pt")

data = torch.load(r'D:\2023-Time\TSCC\Data\KtoC\predict_Czech.pt')
print(data)
"""