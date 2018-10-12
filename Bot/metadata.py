import os
from os.path import isfile, join

os.chdir('../Data/Training Data/')

negative_files = ['Negative/' + f for f in os.listdir('Negative/') if isfile(join('Negative/', f))]
non_negative_files = ['Non-negative/' + f for f in os.listdir('Non-negative/') if isfile(join('Non-negative/', f))]

numWords = []
for n in negative_files:
    with open(n, 'r', encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Negative files finished')

for nn in non_negative_files:
    with open(nn, 'r', encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Non-negative files finished')

numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))
print('The average number of words in the files is', sum(numWords) / len(numWords))

'''
The total number of files is 313217
The total number of words in the files is 22134493
The average number of words in the files is 70.66823639840749
'''