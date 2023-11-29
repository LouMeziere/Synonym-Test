


import gensim.downloader as api
import pandas as pd

wv = api.load('word2vec-google-news-300')

wv.similarity('lexicon','person')

df_synonym = pd.read_csv('synonym.csv')
print(df_synonym.columns)

df_synonym.head(5)

df_synonym['question'][0]

def compute_max(model,target,word1,word2,word3):
    try:
        current_max = model.similarity(word1, target)
        word = word1

        if current_max < model.similarity(word2, target):
            current_max = model.similarity(word2, target)
            word = word2
        if current_max < model.similarity(word3, target):
            current_max = model.similarity(word3, target)
            word = word3
        return word
    except KeyError as e:
        print(f"KeyError: {e}")
        return ''
def find_cosine_similarity(name,model):
    df_synonym = pd.read_csv('synonym.csv')

    file = open(str(name)+'-details.csv','w+')
    for index in range(len(df_synonym)):
        prediction = compute_max(model,str(df_synonym['question'][index]),str(df_synonym['0'][index]),str(df_synonym['1'][index]),str(df_synonym['2'][index]))
        if prediction == df_synonym['answer'][index]:
            file.write(df_synonym['question'][index]+', '+df_synonym['answer'][index]+', '+prediction+', correct')
        else:
            file.write(df_synonym['question'][index]+', '+df_synonym['answer'][index]+', '+prediction+', wrong')
        file.write('\n')
    file.close()
#find_cosine_similarity(wv)

def write_analysis(name,model):
    fil = open(str(name)+'-details.csv','r+')
    lines = fil.readlines()
    right = 0
    for i in range(len(lines)):
        cor_or_wron = lines[i].split(', ')[-1].strip()
        if cor_or_wron == 'correct':
            right+= 1
    new_file = open(str(name)+'_analysis.csv','w+')
    model_name = str(name)
    size_of_corpus = str(len(model))
    size_of_vocab = str(len(df_synonym))
    guesses = len(lines)
    correct = str(right/guesses)

    new_file.write(model_name+', '+size_of_corpus+', '+size_of_vocab+', '+guesses+', '+correct)


    new_file.close()