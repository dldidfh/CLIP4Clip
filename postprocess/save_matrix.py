import os 
import numpy as np 
import pandas as pd 
def save_matrix(args, video_ids, sim_matrix,activity_list=[], batch_sentences=None):
    if activity_list != None:
        activity_list = np.array(activity_list).reshape(-1)
        batch_sentences = np.array(batch_sentences).reshape(-1)
    columns = [id for id in np.array(video_ids).reshape(-1)]
    df = pd.DataFrame(data=sim_matrix, columns=columns)
    if len(activity_list) > 0 :
        df['activity'] = activity_list
        df['sentence'] = batch_sentences
        
        new_col1 = df.columns[-2:].to_list()
        new_col2 = df.columns[:-2].to_list()
        df = df[new_col1+new_col2]
    df.to_csv(f"{os.path.join(args.output_dir, 'sim_matrix.csv')}", index=False)
    # df.to_excel(f"{os.path.join(args.output_dir, 'sim_matrix.xlsx')}", index=False)


