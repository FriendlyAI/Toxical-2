import os

import pandas as pd

os.chdir('../Data/Training Data/Raw')


def encode(line):
    return line.encode('utf-8', errors='replace').decode('utf-8')


def toxicity():
    num = 1

    comments = pd.read_csv('Toxicity/toxicity_annotated_comments.tsv', sep='\t', index_col=0)
    annotations = pd.read_csv('Toxicity/toxicity_annotations.tsv', sep='\t')

    total = len((annotations['rev_id'].unique()))

    labels = annotations.groupby('rev_id')['toxicity_score'].mean() < 0

    comments['toxic'] = labels

    comments['comment'] = comments['comment'].apply(lambda x: x.replace('NEWLINE_TOKEN', ' '))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace('TAB_TOKEN', ' '))

    for index, row in comments.iterrows():
        print(f'{num}/{total}')
        num += 1
        if row['toxic']:
            with open(f'/Users/MacBook/Documents/LSTM Data/Negative/{str(index)}.txt', 'w+') as f:
                f.write(encode(row['comment']))
        else:
            with open(f'/Users/MacBook/Documents/LSTM Data/Non-negative/{str(index)}.txt', 'w+') as f:
                f.write(encode(row['comment']))


def aggression():
    num = 1

    comments = pd.read_csv('Aggression/aggression_annotated_comments.tsv', sep='\t', index_col=0)
    annotations = pd.read_csv('Aggression/aggression_annotations.tsv', sep='\t')

    # total number of comments
    total = len((annotations['rev_id'].unique()))

    # if most users agree that the comment is negative, mark as negative
    labels = annotations.groupby('rev_id')['aggression_score'].mean() < 0
    comments['aggressive'] = labels

    comments['comment'] = comments['comment'].apply(lambda x: x.replace('NEWLINE_TOKEN', ' '))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace('TAB_TOKEN', ' '))

    for index, row in comments.iterrows():
        print(f'{num}/{total}')
        num += 1
        if row['aggressive']:
            with open(f'/Users/MacBook/Documents/LSTM Data/Negative/{str(index)}.txt', 'w+') as f:
                f.write(encode(row['comment']))
        else:
            with open(f'/Users/MacBook/Documents/LSTM Data/Non-negative/{str(index)}.txt', 'w+') as f:
                f.write(encode(row['comment']))


def attack():
    num = 1

    comments = pd.read_csv('Personal Attacks/attack_annotated_comments.tsv', sep='\t', index_col=0)
    annotations = pd.read_csv('Personal Attacks/attack_annotations.tsv', sep='\t')

    # total number of comments
    total = len((annotations['rev_id'].unique()))

    # if most users agree that the comment is negative, mark as negative
    labels = annotations.groupby('rev_id')['attack'].mean() > 0.5
    comments['attack'] = labels

    comments['comment'] = comments['comment'].apply(lambda x: x.replace('NEWLINE_TOKEN', ' '))
    comments['comment'] = comments['comment'].apply(lambda x: x.replace('TAB_TOKEN', ' '))

    for index, row in comments.iterrows():
        print(f'{num}/{total}')
        num += 1
        if row['attack']:
            with open(f'/Users/MacBook/Documents/LSTM Data/Negative/{str(index)}.txt', 'w+') as f:
                f.write(encode(row['comment']))
        else:
            with open(f'/Users/MacBook/Documents/LSTM Data/Non-negative/{str(index)}.txt', 'w+') as f:
                f.write(encode(row['comment']))


toxicity()
aggression()
attack()
