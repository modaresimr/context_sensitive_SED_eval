import sys
import os
import pandas as pd
import numpy as np

rootFolder='data/strong/'
s='eval'
filter_classes=['/m/032s66','/m/07yv9']
gt=pd.read_csv(f'{rootFolder}/audioset_{s}_strong.tsv',delimiter='\t')
gt=gt.loc[gt['label'].isin(filter_classes)]
class_names=pd.read_csv(f'{rootFolder}/mid_to_display_name.tsv',names=['mid','label'],delimiter='\t').set_index('mid').to_dict()['label']
gt['label_full']=gt['label'].map(class_names)

gt=gt[['segment_id','start_time_seconds','end_time_seconds','label_full']]

def eval(chunk):
    from YamnetEval import YamnetEval
    yamnet=YamnetEval()
    yamnet.load()
    all={'gt':[],'pred':[],'failed':[]}
    for f in chunk:
        file=f.split('.')[0]
        try:
            gtt=gt.loc[gt['segment_id']==file]
            
            pred=yamnet.process(f'{rootFolder}{s}_m4a/{f}')
            pred.insert(0,'segment_id',file)

            for r in gtt.values:
                all['gt'].append(r)
            for p in pred.values:
                all['pred'].append(p)
        except:
            all['failed'].append(f)
    return all

from joblib import Parallel, delayed

files=os.listdir(f'{rootFolder}/{s}_m4a/')
chunks=np.array_split(files, 8)
# result=eval(chunks[1])
result=Parallel(n_jobs=8,backend="multiprocessing")(delayed(eval)(c) for c in chunks)
result2={k: v for d in result for k, v in d.items()}
final_gt=[]
final_pred=[]
final_failed=[]
for k in result:
    final_gt+=k['gt']
    final_pred+=k['pred']
    final_failed+=k['failed']

df_real=pd.DataFrame(final_gt,columns=["filename","onset","offset","event_label"])
df_pred=pd.DataFrame(final_pred,columns=["filename","onset","offset","event_label",'p'])

df_real=df_real.sort_values(by=["filename",'onset', 'offset']).drop_duplicates(subset=["filename","onset","offset","event_label"], keep='last')
df_pred=df_pred.sort_values(by=["filename",'onset', 'offset']).drop_duplicates(subset=["filename","onset","offset","event_label"], keep='last')

#######
dir=f'{rootFolder}/{s}'
if not (os.path.exists(dir)):
    os.makedirs(dir)
df_real.to_csv(path_or_buf=f'{dir}/real.tsv',sep='\t', index=False, header=True)
df_pred.to_csv(path_or_buf=f'{dir}/pred.tsv',sep='\t', index=False, header=True)
np.savetxt(f'{dir}/failed.txt',final_failed)
#if __name__ == '__main__':
#
