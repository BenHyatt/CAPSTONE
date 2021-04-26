# -*- coding: utf-8 -*-
#Created by Ben Underwood for Capstone 2021
"""
Spyder Editor

"""
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim.downloader
from termcolor import colored
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#Contains various obscure countries and their capitals
country_capital_dic={
    'Uganda': 'Kampala',
    'Gambia': 'Banjul',
    'Ghana': 'Accra' ,  
    'Libya': 'Tripoli',
    'Liberia':'Monrovia',
    'Malta':'Valletta',
    'Mauritania':'Nouakchott',
    #'England':'London'
    #'Mali': 'Bamako'
    #'Serbia': 'Belgrade',
    #'Latvia' : 'Riga'
}


#Contains states and their capitals as key/value pairs
capital_dic={
    'Alabama': 'Montgomery',
    'Alaska': 'Juneau',
    'Arizona':'Phoenix',
    'Arkansas':'Little Rock',
    'California': 'Sacramento',
    'Colorado':'Denver',
    'Connecticut':'Hartford',
    'Delaware':'Dover',
    'Florida': 'Tallahassee',
    'Georgia': 'Atlanta',
    'Hawaii': 'Honolulu',
    'Idaho': 'Boise',
    'Illinois': 'Springfield',
    'Indiana': 'Indianapolis',
    'Iowa': 'Des Moines',
    'Kansas': 'Topeka',
    'Kentucky': 'Frankfort',
    'Louisiana': 'Baton Rouge',
    'Maine': 'Augusta',
    'Maryland': 'Annapolis',
    'Massachusetts': 'Boston',
    'Michigan': 'Lansing',
    'Minnesota': 'St. Paul',
    'Mississippi': 'Jackson',
    'Missouri': 'Jefferson City',
    'Montana': 'Helena',
    'Nebraska': 'Lincoln',
    'Nevada': 'Carson City',
    'New Hampshire': 'Concord',
    'New Jersey': 'Trenton',
    'New Mexico': 'Santa Fe',
    'New York': 'Albany',
    'North Carolina': 'Raleigh',
    'North Dakota': 'Bismarck',
    'Ohio': 'Columbus',
    'Oklahoma': 'Oklahoma City',
    'Oregon': 'Salem',
    'Pennsylvania': 'Harrisburg',
    'Rhode Island': 'Providence',
    'South Carolina': 'Columbia',
    'South Dakoda': 'Pierre',
    'Tennessee': 'Nashville',
    'Texas': 'Austin',
    'Utah': 'Salt Lake City',
    'Vermont': 'Montpelier',
    'Virginia': 'Richmond',
    'Washington': 'Olympia',
    'West Virginia': 'Charleston',
    'Wisconsin': 'Madison',
    'Wyoming': 'Cheyenne'  
}

#Returns the top word that solves (returns d in) the following analogy given a,b,c a:b::c:d
def analogy(wordOne, wordTwo, wordThree):
    return model.most_similar(negative=[wordOne], positive=[wordTwo, wordThree],topn=1)[0][0]
    
#Gives the top arbitrary number of words to an arbitrary degree
def analogyN(wordOne, wordTwo, wordThree,n):
    array=model.most_similar(negative=[wordOne], 
                                positive=[wordTwo, wordThree],topn=n)
    ret=array[0][0]
    for i in range(1,n):
        ret=ret + ", " + array[i][0]
    return ret

#Uses the vector definitions of words and returns top Five
#Not used
def vectorAnalogy(wordOne,wordTwo,wordThree):
    return model.most_similar(positive=[model[wordTwo]-model[wordOne]+model[wordThree]],topn=5)

#Helper method
def numWords(word):
    return len(word.split())

model = gensim.downloader.load("fasttext-wiki-news-subwords-300")  # download the model and return as object ready for use
model.save("wiki-news")


#glove = gensim.downloader.load("glove-wiki-gigaword-300")

#googleN = gensim.downloader.load("googleModel")
#googleN.save("googleModel")
model.load("googleModel")

highestState=""
highest=0;
for state in capital_dic:
    if(numWords(state)==1 and numWords(capital_dic[state])==1):
        #print(colored("RUNNING STATE CAPITAL TEST ON " + state,"green","on_red"))
        correctCnt=0
        n=1
        for key in capital_dic:
            if(numWords(key)==1 and numWords(capital_dic[key])==1):
                modelAns=analogy(state,capital_dic[state],key)
                msg=colored("Incorrect","red")
                if(modelAns == capital_dic[key]):
                    msg=colored("Correct  ","green")
                    correctCnt=correctCnt+1
                #print(str(n) + "\t" + msg + "\t" + key + "\t\"" + modelAns + "\"")
                n=n+1
        if(correctCnt>highest):
            highestState=state
            highest=correctCnt
        print(state + " accuracy: " + str(correctCnt) + "/" + str(n) + " = " + str(round(100*correctCnt/n)) + "%")

#
# Verbose Ohio Capital Test
#
correctCnt=0
n=1
for key in capital_dic:
    if(numWords(key)==1 and numWords(capital_dic[key])==1 and key!="Ohio"):
        modelAns=analogy("Ohio","Columbus",key)
        msg=colored("Incorrect","red")
        if(modelAns==capital_dic[key]):
            msg=colored("Correct  ","green")
            correctCnt=correctCnt+1
            print(str(n) + "\t" + str(msg) + "\t" + str(key) + "\t\"" + str(modelAns) + "\"")
        else:
            for i in range(0,10):
                if(model.most_similar(topn=10,positive=["Columbus",key], negative=["Ohio"])[i][0]==capital_dic[key]):
                    print(str(n) + "\t" + str(msg) + "\t" + str(key) + "\t\"" + str(modelAns) + "\", but correct on try " + str(i+1) + ".")
                    break
                elif(i==9):
                    print(str(n) + "\t" + str(msg) + "\t" + str(key) + "\t\"" + str(modelAns) + "\"")
        n=n+1
print("Accuracy: " + str(correctCnt) + "/" + str(n) + " = " + str(round(100*correctCnt/n)) + "%")





print(list(gensim.downloader.info()['models'].keys()))
corpus = gensim.downloader.load('text8')  # download the corpus and return it opened as an iterable
model = Word2Vec(corpus)  # train a model from the corpus
model.wv.most_similar("car")
model.train



import re
import nltk
nltk.download('punkt')
print(common_texts)
model = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
model.save("word2vec.model")
model = Word2Vec.load("word2vec.model")
model.train([["hello", "world"]], total_examples=1, epochs=1)
vector = model.wv['computer']  # get numpy vector of a word
sims = model.wv.most_similar('shocked', topn=10)  # get other similar words
print(sims)
print(vector)

text="Word embedding is the collective name for a set of language modeling and feature learning techniques in natural language processing where words or phrases from the vocabulary are mapped to vectors of real numbers. Conceptually it involves a mathematical embedding from a space with many dimensions per word to a continuous vector space with a much lower dimension.The use of multi-sense embeddings is known to improve performance in several NLP tasks, such as part-of-speech tagging, semantic relation identification, and semantic relatedness. However, tasks involving named entity recognition and sentiment analysis seem not to benefit from a multiple vector representation."

text=re.sub(r"[^.A-Za-z]",' ',text)
text
sentence=text.split('.')
sentence
tokens=[nltk.word_tokenize(words) for words in sentence]
tokens
model = Word2Vec(sentences=tokens,vector_size=50,window=5,sg=1,min_count=1)
model.wv["the"]

#Return 2d data of large dimentional vectors
def PCA_reduction(words):
    return PCA(n_components=2).fit_transform(words)[:,:2]

def TSNE_reduction(words):
    return TSNE(random_state=0, perplexity = 3, learning_rate = 1000, n_iter = 10000).fit_transform(words)[:,:2]


#Plots a 2D scatterplot of words using either PCA or TSNE
#Can set a custome label
def display_scatterplot(model, words,type="pca", label="",lines=False):
    word_vectors = np.array([model[w] for w in words])
    if(type == "tsne"):
        twodim = TSNE_reduction(word_vectors)  
    else:
        print("PCA")
        twodim=PCA_reduction(word_vectors)
    plt.figure(figsize=(6,6),dpi=600)
    hfont = {'fontname':'Helvetica'}
    if(type=="tsne"):
        plt.title("Visualization of word vectors through t-SNE", **hfont)
    else:
        #plt.title("PCA Dimensional Reduction of Country/Capital Word Pair Relationships", **hfont)
        plt.title("Visualization of Country Vectors through PCA", **hfont)
    if(not label==""):
        plt.title(label,**hfont)
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='w') #k is black (b -> blue)
    plt.xlabel("Underwood, 2021", **hfont,loc='center')
    
    #Remove tick labels
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])
    
    #Add grid lines
    plt.gca().axes.grid(linestyle='--')
    
    #Add labels for the words using "proper" distances
    for word, (x,y) in zip(words, twodim):
        if(type=="tsne"):
            plt.text(x+7, y+7, word,**hfont)
        else:
            plt.text(x+.005,y+.005,word,**hfont)
    
    #Draw lines between word projections that are adjacent in the words matrix
    if(lines):
        for i in range(0,len(words),2):
            plt.plot([twodim[i,0],twodim[i+1,0]],[twodim[i,1],twodim[i+1,1]],color='k')
    
    plt.show()

#Plots a non-interactive 3D projection
def display_pca_scatterplot3D(model, words):
    word_vectors = np.array([model[w] for w in words])
    pca=PCA(n_components=3)
    threeDim= pca.fit_transform(word_vectors)
    print(threeDim)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(threeDim[:,0],threeDim[:,1],threeDim[:,2])


#Can be used to plot a scatterplot of various words
words=[]
print(words)
n=0
for state in capital_dic:
    if(numWords(state)==1 and numWords(capital_dic[state])==1 and n>20 and n<35):
        words.append(state)
        words.append(capital_dic[state])
    n=n+1
print(words)
display_scatterplot(model, words)



display_scatterplot(model,["Obama","Bush","Clinton","Carter","Reagan","Ford","Nixon","Kennedy"])
display_pca_scatterplot3D(model,["ugly","wretched","nice","beautiful"])


print(analogy("England","London","Ohio"))
print("England:London::Canada:???")
random_countries={
    'Indonesia',
    'Pakistan',
    'Ireland',
    'Australia',
    'Portugal'    
}
print(analogy("Canada","Ottawa","Portugal"))
for i in range(0,5):
    print(str(i+1) + ": " + model.most_similar(positive=["Canada","Ottowa"],negative=["Ohio"])[i][0])

print("England:")

#Countries graphic
words=[]
for country in country_capital_dic:
    print(analogy("England","London",country)+ " " + country_capital_dic[country])
    words.append(country)
    words.append(country_capital_dic[country])
display_scatterplot(model,words)
display_scatterplot(model,["Spain","Germany","Canada","Ghana","Gambia","Mali","Uganda","Ethiopia","Egypt","Turkey","Senegal"])
display_scatterplot(model,["outstanding","beautiful","gorgeous","supreme","magnificent","attractive","superb","hideous","horrible","grotesque","ugly","horrendous","ghastly","revolting"])
model["vector"]

model.similar_by_word("antibiased")
model.least_similar("good")


#Display heatmap
words=["vector"]
#words=["magnificent","outstanding","splendid","horrid","hideous","ghastly"]
word_vectors = np.array([model[w] for w in words])
fig, ax = plt.subplots(figsize=(6,3),dpi=1000)
im = ax.imshow(word_vectors, aspect='auto')
ax.set_yticks(np.arange(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel("Underwood, 2021")
ax.set_title("Color-coded Word Vector of \"vector\"")
#minor_locator = AutoMinorLocator(2)
#plt.gca().yaxis.set_minor_locator(minor_locator)
#plt.grid(axis = 'y',color='w',which='minor',linewidth=2)
fig.tight_layout()
plt.show()

model.similarity("hideous","ghastly")

adjectives=["car","cars","leaf","leaves","knife","knives","table","tables","bag","bags","truck","trucks"]
display_scatterplot(model,adjectives)





#Develop a thesaurus 
import pandas as pd
df = pd.read_excel('Documents/CAPSTONE/words.xlsx') #Open the excel document
adjectives = df['personality'] #Select the correct column
thesaurus=[]
for i in range(len(adjectives)):
    temp=[adjectives[i].title()]
    #Select top 20 as post-processing my lower down to 10 or fewer
    synonyms=model.most_similar(adjectives[i],topn=20)
    cnt=0
    for n in range(20):
        #Don't include synonyms that contain the headword
        if(not(adjectives[i] in synonyms[n][0].lower()) and cnt<10):
            temp.append(synonyms[n][0].lower())
            cnt=cnt+1
    thesaurus.append(temp)

#Save the thesaurus array into a spreadsheet
df = pd.DataFrame(thesaurus)
df.to_excel(excel_writer = "Documents/CAPSTONE/thesaurus.xlsx")



#Random analogie and code
word="funny"
syn=model.most_similar(word)
for i in range(10):
    print(str(i+1) + ":\t" + syn[i][0])

print(analogyN("Facebook","Zuckerberg","Amazon",5))
syn=model.most_similar("bossy",topn=10)
for i in range(10):
    print(syn[i][0])

w=['man','woman','leader','cleaner','doctor','nurse','actor','actress']
display_scatterplot(model,w)



print("England:London :: Vietnam:" + analogy("England","London","Vietnam")) 
print(analogyN("England","London","Australia",5))   

