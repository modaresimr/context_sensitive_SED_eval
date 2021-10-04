import pandas as pd
import numpy as np

def merge(df):
    delta = .1
    newdf = []
    lastFile = None
    last = {c: None for c in df['event_label'].unique()}
    for k, row in df.iterrows():
        file = row['filename']
        label = row['event_label']
        if(lastFile != file):
            for c in last:
                if not(last[c] is None):
                    newdf.append(last[c])
                last[c] = None

        lastFile = file
        if(last[label] is None):
            last[label] = row
        else:
            if(last[label]['offset'] >= row['onset']-delta):
                last[label]['offset'] = row['offset']
            else:
                newdf.append(last[label])
                last[label] = row

    for c in last:
        if not(last[c] is None):
            newdf.append(last[c])
    return pd.DataFrame(newdf)

def merge_array(arr):
    if len(arr) == 0:
        return arr
    delta = .1
    newdf = []
    for  row in arr:
        if len(newdf) > 0 and newdf[-1][1] >= row[0]-delta:
            newdf[-1][1]=row[1]
        else:
            newdf.append(row)
    newdf= np.array(newdf)
    
    if len(newdf.shape)==1:
        newdf=newdf.reshape(1,2)
    return newdf

if __name__ == '__main__':
    pef = '../data/strong//eval/pred.tsv'
    pred = merge(pd.read_csv(pef, delimiter='\t'))
