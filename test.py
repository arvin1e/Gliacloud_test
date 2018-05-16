import math
import nltk
from nltk.collocations import *

def ngram_probs(filename='English_artical.txt'):
    line = ""
    a = 'we'
    b = 'are'
    c = 'family'
    cnt2 = cnt3 = 0
    open_file = open(filename,'r')
    for str1 in open_file:
        line += str1.lower()
    tokens = line.split()
    bgs = nltk.bigrams(tokens)
    tgs = nltk.trigrams(tokens)
    bdist = nltk.FreqDist(bgs)
    tdist = nltk.FreqDist(tgs)
    
    for i,j in bdist.items():
        if i[0]==a and i[1]==b:
            print(i,j)
            cnt2 = j
            break
    for i,j in tdist.items():
        if i[0]==a and i[1]==b and i[2]==c:
            print(i,j)
            cnt3 = j
            break
    return cnt2,cnt3

def prob3(cnt2, cnt3):
    prob = math.log(cnt3/cnt2)
    return prob

if __name__=="__main__":
    cnt2,cnt3 = ngram_probs()
    print(cnt2,cnt3)
    p = prob3(cnt2,cnt3)
    print(p)
    
    '''
    #Q2-3
    use ngram_probs-similar function without break to get a list of (bigram,cnt),
    choose the max likelihood one(cnt I thought) as next word 
    and further going until containing '.' in bigram or length of sentence exceed 15.
    #Q2-4 unknown
    '''
    
    
    
    
    
    
    
    
    
    
    