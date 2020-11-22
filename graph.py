import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

a=[[2256,  781, 1702],
 [ 253 , 950 , 484],
 [  48  , 50 , 962]]
b=[[4518,  125,   96],
 [ 325 ,1293  , 69],
 [ 140  , 35 , 885]]


nb = pd.DataFrame(a, index=["Positive", "Negative", "Neutral"],columns=["Positive", "Negative", "Neutral"])
rf = pd.DataFrame(b, index=["Positive", "Negative", "Neutral"],columns=["Positive", "Negative", "Neutral"])

sn.set(font_scale=1.4)#for label size
sn.heatmap(nb, annot=True,annot_kws={"size": 16})# font size
sn.heatmap(rf, annot=True,annot_kws={"size": 16})# font size

