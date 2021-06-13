import pandas as pd
import copy

# seq : log_id list
# n : n-gram
def seq2gram(seq, n):  
    result = []
    tmp = copy.deepcopy(seq)
    seq_len = len(seq)
    for i in range(0, n-1):
        tmp.append("0")

    for i in range(0, seq_len):
        result.append("a".join(tmp[i:i+n]))

    return result


def labeling(row, label_list):
    if row['player_id'] in label_list:
        return "rolling"
    else:
        return "normal"

def preprocessing():
    # n-gram
    n = 3

    # read data
    # rawdata = pd.read_csv("./data/sorted.csv", names=['player_id', 'timestamp', 'log_id'])
    rawdata = pd.read_csv("G:/dataset_csv/result_uid_p1.csv", names=['player_id', 'timestamp', 'log_id'])

    # label = pd.read_csv("./data/label.csv")
    label = pd.read_csv("./data/label_uid.csv")

    

    rawdata = rawdata.dropna(axis=0).reset_index(drop=True)
    rawdata.log_id = rawdata.log_id.astype(int)
    rawdata.log_id = rawdata.log_id.astype(str)

    # gen sequence
    seqdata = rawdata.groupby('player_id')['log_id'].apply(list).reset_index(name='log_seq')
    seqdata['log_seq_len'] = seqdata.apply(lambda row: len(row['log_seq']), axis=1)


    seqdata['log_ngram_seq'] = seqdata.apply(lambda row: seq2gram(row['log_seq'], n), axis=1)
    del seqdata['log_seq']

    # add label
    label_list = label['player_id'].tolist()

    seqdata['label'] = seqdata.apply(lambda row: labeling(row, label_list), axis=1)

    # save preprocessed dataset
    print(seqdata.head())
    print(seqdata.info())
    seqdata.to_json("./data/preprocessed_data_" + str(n) + "gram_uid_p1.json", orient='records')
    seqdata.to_csv("./data/preprocessed_data_" + str(n) + "gram_uid_p1.csv")

    return


preprocessing()

