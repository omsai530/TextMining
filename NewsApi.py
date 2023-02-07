#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:48:01 2023

@author: omsai
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 00:35:08 2023

@author: omsai
"""

########################################
## 2021 - Gates
##
## Topics: 
    # Data gathering via API
    #  - URLs and GET
    # Cleaning and preparing text DATA
    # DTM and Data Frames
    # Training and Testing at DT
    # CLustering
    ## LDA
    
#########################################    
    
    
## ATTENTION READER...
##
## First, you will need to go to 
## https://newsapi.org/
## https://newsapi.org/register
## and get an API key

 

################## DO NOT USE MY KEY!!
## Get your own key. 
##
###################################################


### API KEY  - get a key!
##https://newsapi.org/

## Example URL
## https://newsapi.org/v2/everything?
## q=tesla&from=2021-05-20&sortBy=publishedAt&
## apiKey=YOUR KEY HERE


## What to import
import requests  ## for getting data from a server GET
import re   ## for regular expressions
import pandas as pd    ## for dataframes and related
from pandas import DataFrame

## To tokenize and vectorize text type data
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
## For word clouds
## conda install -c conda-forge wordcloud
## May also have to run conda update --all on cmd
#import PIL
#import Pillow
#import wordcloud
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
#from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import tree
## conda install python-graphviz
## restart kernel (click the little red x next to the Console)
import graphviz

from sklearn.decomposition import LatentDirichletAllocation 
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import silhouette_samples, silhouette_score
import sklearn
from sklearn.cluster import KMeans

from sklearn import preprocessing

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram


####################################
##
##  Step 1: Connect to the server
##          Send a query
##          Collect and clean the 
##          results
####################################

####################################################
##In the following loop, we will query thenewsapi servers
##for all the topic names in the list
## We will then build a large csv file 
## where each article is a row
##
## From there, we will convert this data
## into a labeled dataframe
## so we can train and then test our DT
## model
####################################################

####################################################
## Build the URL and GET the results
## NOTE: At the bottom of this code
## commented out, you will find a second
## method for doing the following. This is FYI.
####################################################

## This is the endpoint - the server and 
## location on the server where your data 
## will be retrieved from

## TEST FIRST!
## We are about to build this URL:
## https://newsapi.org/v2/everything?apiKey=8f4134f7d0de43b8b49f91e22100f22b&q=bitcoin

 

topics=[ 'recision', 'inflation', 'oil prices', 'War', 'stock market']


 


## topics needs to be a list of strings (words)
## Next, let's build the csv file
## first and add the column names
## Create a new csv file to save the headlines
filename="NewHeadlines.csv"
MyFILE=open(filename,"w")  # "a"  for append   "r" for read
## with open
### Place the column names in - write to the first row
WriteThis="LABEL,Date,Source,Title,Headline\n"
MyFILE.write(WriteThis)
MyFILE.close()

## CHeck it! Can you find this file?
    
#### --------------------> GATHER - CLEAN - CREATE FILE    

## RE: documentation and options
## https://newsapi.org/docs/endpoints/everything

endpoint="https://newsapi.org/v2/everything"

################# enter for loop to collect
################# data on three topics
#######################################

for topic in topics:

    ## Dictionary Structure
    URLPost = {'apiKey':'8f4134f7d0de43b8b49f91e22100f22b',
               'q':topic,
               'language':'en'
    }

    response=requests.get(endpoint, URLPost)
    print(response)
    jsontxt = response.json()
    print(jsontxt)
    #####################################################
    
    
    ## Open the file for append
    MyFILE=open(filename, "a")
    LABEL=topic
    for items in jsontxt["articles"]:
        print(items, "\n\n\n")
                  
        #Author=items["author"]
        #Author=str(Author)
        #Author=Author.replace(',', '')
        
        Source=items["source"]["name"]
        print(Source)
        
        Date=items["publishedAt"]
        ##clean up the date
        NewDate=Date.split("T")
        Date=NewDate[0]
        print(Date)
        
        ## CLEAN the Title
        ##----------------------------------------------------------
        ##Replace punctuation with space
        # Accept one or more copies of punctuation         
        # plus zero or more copies of a space
        # and replace it with a single space
        Title=items["title"]
        Title=str(Title)
        #print(Title)
        Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(' +', ' ', str(Title), flags=re.IGNORECASE)
        Title=re.sub(r'\"', ' ', str(Title), flags=re.IGNORECASE)
        
        # and replace it with a single space
        ## NOTE: Using the "^" on the inside of the [] means
        ## we want to look for any chars NOT a-z or A-Z and replace
        ## them with blank. This removes chars that should not be there.
        Title=re.sub(r'[^a-zA-Z]', " ", str(Title), flags=re.VERBOSE)
        Title=Title.replace(',', '')
        Title=' '.join(Title.split())
        Title=re.sub("\n|\r", "", Title)
        print(Title)
        ##----------------------------------------------------------
        
        Headline=items["description"]
        Headline=str(Headline)
        Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(' +', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
        Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)
        ## Be sure there are no commas in the headlines or it will
        ## write poorly to a csv file....
        Headline=Headline.replace(',', '')
        Headline=' '.join(Headline.split())
        Headline=re.sub("\n|\r", "", Headline)
        
        ### AS AN OPTION - remove words of a given length............
        Headline = ' '.join([wd for wd in Headline.split() if len(wd)>3])
    
        #print("Author: ", Author, "\n")
        #print("Title: ", Title, "\n")
        #print("Headline News Item: ", Headline, "\n\n")
        
        #print(Author)
        print(Title)
        print(Headline)
        
        WriteThis=str(LABEL)+","+str(Date)+","+str(Source)+","+ str(Title) + "," + str(Headline) + "\n"
        print(WriteThis)
        
        MyFILE.write(WriteThis)
        
    ## CLOSE THE FILE
    MyFILE.close()
    
################## END for loop

####################################################
##
## Where are we now?
## 
## So far, we have created a csv file
## with labeled data. Each row is a news article
##
## - BUT - 
## We are not done. We need to choose which
## parts of this data to use to model our decision tree
## and we need to convert the data into a data frame.
##
########################################################


BBC_DF=pd.read_csv(filename, error_bad_lines=False)
print(BBC_DF.head())
# iterating the columns 
for col in BBC_DF.columns: 
    print(col) 
    
print(BBC_DF["Headline"])

## REMOVE any rows with NaN in them
BBC_DF = BBC_DF.dropna()
print(BBC_DF["Headline"])

### Tokenize and Vectorize the Headlines
## Create the list of headlines
## Keep the labels!

HeadlineLIST=[]
LabelLIST=[]

for nexthead, nextlabel in zip(BBC_DF["Headline"], BBC_DF["LABEL"]):
    HeadlineLIST.append(nexthead)
    LabelLIST.append(nextlabel)

print("The headline list is:\n")
print(HeadlineLIST)

print("The label list is:\n")
print(LabelLIST)


##########################################
## Remove all words that match the topics.
## For example, if the topics are food and covid
## remove these exact words.
##
## We will need to do this by hand. 
NewHeadlineLIST=[]

for element in HeadlineLIST:
    print(element)
    print(type(element))
    ## make into list
    AllWords=element.split(" ")
    print(AllWords)
    
    ## Now remove words that are in your topics
    NewWordsList=[]
    for word in AllWords:
        print(word)
        word=word.lower()
        if word in topics:
            print(word)
        else:
            NewWordsList.append(word)
            
    ##turn back to string
    NewWords=" ".join(NewWordsList)
    ## Place into NewHeadlineLIST
    NewHeadlineLIST.append(NewWords)


##
## Set the     HeadlineLIST to the new one
HeadlineLIST=NewHeadlineLIST
print(HeadlineLIST)     
#########################################
##
##  Build the labeled dataframe
##
######################################################


### Vectorize
## Instantiate your CV
MyCountV=CountVectorizer(
        input="content",  ## because we have a csv file
        lowercase=True, 
        stop_words = "english"
        )

## Use your CV 
MyDTM = MyCountV.fit_transform(HeadlineLIST)  # create a sparse matrix
MyDTM
print(type(MyDTM))


ColumnNames=MyCountV.get_feature_names()
#print(type(ColumnNames))


## Build the data frame
MyDTM_DF=pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)

## Convert the labels from list to df
Labels_DF = DataFrame(LabelLIST,columns=['LABEL'])

## Check your new DF and you new Labels df:
print("Labels\n")
print(Labels_DF)
print("News df\n")
print(MyDTM_DF.iloc[:,0:6])

##Save original DF - without the lables
My_Orig_DF=MyDTM_DF
print(My_Orig_DF)
######################
## AND - just to make sure our dataframe is fair
## let's remove columns called:
## food, bitcoin, and sports (as these are label names)
######################
#MyDTM_DF=MyDTM_DF.drop(topics, axis=1)


## Now - let's create a complete and labeled
## dataframe:
dfs = [Labels_DF, MyDTM_DF]
print(dfs)

Final_News_DF_Labeled = pd.concat(dfs,axis=1, join='inner')
## DF with labels
print(Final_News_DF_Labeled)
Final_News_DF_Labeled.to_csv("War_Consequenses.csv")
List_of_WC=[]

for mytopic in topics:

    tempdf = Final_News_DF_Labeled[Final_News_DF_Labeled['LABEL'] == mytopic]
    print(tempdf)
    
    tempdf =tempdf.sum(axis=0,numeric_only=True)
    #print(tempdf)
    
    #Make var name
    NextVarName=str("wc"+str(mytopic))
    #print( NextVarName)
    
    ##In the same folder as this code, I have three images
    ## They are called: food.jpg, bitcoin.jpg, and sports.jpg
    #next_image=str(str(mytopic) + ".jpg")
    #print(next_image)
    
    ## https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
    
    ###########
    ## Create and store in a list the wordcloud OBJECTS
    #########
    NextVarName = WordCloud(width=1000, height=600, background_color="white",
                   min_word_length=4, #mask=next_image,
                   max_words=200).generate_from_frequencies(tempdf)
    
    ## Here, this list holds all three wordclouds I am building
    List_of_WC.append(NextVarName)
    

##------------------------------------------------------------------
print(List_of_WC)
##########
########## Create the wordclouds
##########
fig=plt.figure(figsize=(25, 25))
#figure, axes = plt.subplots(nrows=2, ncols=2)
NumTopics=len(topics)
for i in range(NumTopics):
    print(i)
    ax = fig.add_subplot(NumTopics,1,i+1)
    plt.imshow(List_of_WC[i], interpolation='bilinear')
    plt.axis("off")
    plt.savefig("NewClouds.pdf")
 
    
 
###########################################################
##
##
##                  Clustering
##
##
############################################################
## Our DF
print(My_Orig_DF)

#from sklearn.metrics import silhouette_samples, silhouette_score
#from sklearn.cluster import KMeans

My_KMean= KMeans(n_clusters=3)
My_KMean.fit(My_Orig_DF)
My_labels=My_KMean.predict(My_Orig_DF)
print(My_labels)

#from sklearn import preprocessing
#from sklearn.cluster import KMeans
#import seaborn as sns

My_KMean2 = KMeans(n_clusters=4).fit(preprocessing.normalize(My_Orig_DF))
My_KMean2.fit(My_Orig_DF)
My_labels2=My_KMean2.predict(My_Orig_DF)
print(My_labels2)

My_KMean3= KMeans(n_clusters=3)
My_KMean3.fit(My_Orig_DF)
My_labels3=My_KMean3.predict(My_Orig_DF)
print("Silhouette Score for k = 3 \n",silhouette_score(My_Orig_DF, My_labels3))


#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
#length of the document: called cosine similarity
cosdist = 1 - cosine_similarity(MyDTM)
print(cosdist)
print(np.round(cosdist,3))  #cos dist should be .02

#----------------------------------------------------------
## Hierarchical Clustering using ward and cosine sim
linkage_matrix = ward(cosdist) #define the linkage_matrix 
#using ward clustering pre-computed distances
print(linkage_matrix)
fig = plt.figure(figsize=(25, 10))
dn = dendrogram(linkage_matrix)
plt.show()

 

###############################################################
##
##               Model with two ML supervised options
##
##               DT
##               NB (multinomial)
##      
###############################################################         
## STEP 1   Create Training and Testing Data
###############################################################
## Write the dataframe to csv so you can use it later if you wish
##
Final_News_DF_Labeled.to_csv("Labeled_News_Data_from_API.csv")
TrainDF, TestDF = train_test_split(Final_News_DF_Labeled, test_size=0.3)
print(TrainDF)
print(TestDF)

#################################################
## STEP 2: Separate LABELS
#################################################
## IMPORTANT - YOU CANNOT LEAVE LABELS ON 
## Save labels

### TEST ---------------------
TestLabels=TestDF["LABEL"]
print(TestLabels)
TestDF = TestDF.drop(["LABEL"], axis=1)
print(TestDF)
### TRAIN----------------------
TrainLabels=TrainDF["LABEL"]
print(TrainLabels)
## remove labels
TrainDF = TrainDF.drop(["LABEL"], axis=1)

##################################################
## STEP 3:  Run MNB
##################################################

## Instantiate
MyModelNB= MultinomialNB()

## FIT
MyNB=MyModelNB.fit(TrainDF, TrainLabels)
#print(MyNB.classes_)
#print(MyNB.class_count_)
#print(MyNB.feature_log_prob_)


Prediction = MyModelNB.predict(TestDF)
print(np.round(MyModelNB.predict_proba(TestDF),2))

## COnfusion Matrix Accuracies
cnf_matrix = confusion_matrix(TestLabels, Prediction)
print("\nThe confusion matrix is:")
print(cnf_matrix)


##################################################
## STEP 3:  Run DT
##################################################

## Instantiate
MyDT=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)

##
MyDT.fit(TrainDF, TrainLabels)

#tree.plot_tree(MyDT)
#plt.savefig(temp1)

feature_names=TrainDF.columns
Tree_Object = tree.export_graphviz(MyDT, out_file=None,
                    ## The following creates TrainDF.columns for each
                    ## which are the feature names.
                      feature_names=feature_names,  
                      class_names=topics,
                      #["food","sports","bitcoin"],  
                      filled=True, rounded=True,  
                      special_characters=True)      
                              
graph = graphviz.Source(Tree_Object) 
    
graph.render("MyTree") 


## COnfusion Matrix
print("Prediction\n")
DT_pred=MyDT.predict(TestDF)
print(DT_pred)
    
bn_matrix = confusion_matrix(TestLabels, DT_pred)
print("\nThe confusion matrix is:")
print(bn_matrix)


FeatureImp=MyDT.feature_importances_   
indices = np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF.shape[1]):
    if FeatureImp[indices[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices[f], FeatureImp[indices[f]]))
        print ("feature name: ", feature_names[indices[f]])
        
        
        
##############################################
##
##   LDA Topics Modeling
##
##
#########################################################
NUM_TOPICS=5
# = NUM_TOPICS

lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=100, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
   
lda_Z_DF = lda_model.fit_transform(My_Orig_DF)
print(lda_Z_DF.shape)  # (NO_DOCUMENTS, NO_TOPICS)

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                    for i in topic.argsort()[:-top_n - 1:-1]])
 
print("LDA Model:")
print_topics(lda_model, MyCountV)

############## Fancy Plot.................
import warnings
#warnings.filterwarnings("ignore", category=DeprecationWarning)
import pyLDAvis.sklearn as LDAvis
import pyLDAvis

# ## conda install -c conda-forge pyldavis
pyLDAvis.enable_notebook() ## not using notebook
panel = LDAvis.prepare(lda_model, MyDTM, MyCountV,  mds='tsne')
pyLDAvis.save_html(panel, "InTheNews.html")

################ Another fun vis for LDA

word_topic = np.array(lda_model.components_)
#print(word_topic)
word_topic = word_topic.transpose()

num_top_words = 15
vocab_array = np.asarray(ColumnNames)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 14

for t in range(NUM_TOPICS):
    plt.subplot(1, NUM_TOPICS, t + 1)  # plot numbering starts with 1
    plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
    plt.xticks([])  # remove x-axis markings ('ticks')
    plt.yticks([]) # remove y-axis markings ('ticks')
    plt.title('Topic #{}'.format(t))
    top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
    top_words_idx = top_words_idx[:num_top_words]
    top_words = vocab_array[top_words_idx]
    top_words_shares = word_topic[top_words_idx, t]
    for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
        plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
                 ##fontsize_base*share)

#plt.tight_layout()
#plt.show()
plt.savefig("TopicsVis.pdf")

 

#############################################
## Silhouette and clusters
#############################################

#from sklearn.metrics import silhouette_samples, silhouette_score

## Using MyDTM_DF which is not labeled

# =============================================================================
#     KMean= KMeans(n_clusters=3)
#     KMean.fit(MyDTM_DF)
#     label=KMean.predict(MyDTM_DF)
#     print(label)
#     
#     #sns.scatterplot(MyDTM_DF[0],MyDTM_DF[1], hue=label)
#     print("Silhouette Score for k=3\n",silhouette_score(MyDTM_DF, label))
#       #
# =============================================================================

 

##############################
## Check These files now on your computer...
#############################################
## NewClouds.pdf
## TopicsVis.pdf
## InTheNews.html
## MyTree.pdf
## Labeled_News_Data_from_API.csv



    

