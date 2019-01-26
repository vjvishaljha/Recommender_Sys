# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 20:31:14 2019

@author: Vishal
"""
#Importing the required libraries
import pandas as pd 
import numpy as np
import random
from playsound import playsound
import speech_recognition as sr
import re 
import gtts
from surprise import Reader, Dataset, SVD, evaluate

#importing the dataset
movies_dataset=pd.read_csv('ml-latest-small/movies.csv')
ratings_dataset=pd.read_csv('ml-latest-small/ratings.csv')

#Converting the format of Genre column to a list and then appending to the new list
Genre=[]
Genres={}
for num in range(0,len(movies_dataset)):
    key=movies_dataset.iloc[num]['title']
    value=movies_dataset.iloc[num]['genres'].split('|')
    Genres[key]=value
    Genre.append(value)

    
#Making a new column in our original Dataset         
movies_dataset['new'] =Genre

#Getting the year from the movie column 
p = re.compile(r"(?:\((\d{4})\))?\s*$")
years=[]
for movies in movies_dataset['title']:
     m = p.search(movies)
     year = m.group(1)
     years.append(year)  
movies_dataset['year']=years

#Deleting the year from the movies title column
movies_name=[]
raw=[]
for movies in movies_dataset['title']:
     m = p.search(movies)
     year = m.group(0)
     new=re.split(year,movies)
     raw.append(new)  
for i in range(len(raw)):
    movies_name.append(raw[i][0][:-2].title())
    

movies_dataset['movie_name']=movies_name


#Converting the datatype of new column from list to string as required by the function
movies_dataset['new']=movies_dataset['new'].apply(' '.join)

'''Applying the Cotent Based Filtering'''
 #Applying Feature extraction 
from sklearn.feature_extraction.text import TfidfVectorizer

tfid=TfidfVectorizer(stop_words='english')
#matrix after applying the tfidf
matrix=tfid.fit_transform(movies_dataset['new'])

#Compute the cosine similarity of every genre
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim=cosine_similarity(matrix,matrix)

'''Applying the Collaborative Filtering'''
#Intialising the Reader which is used to parse the file containing the ratings 
reader=Reader()

#Making the dataset containing the column as userid itemid ratings
#the order is very specific and we have to follow the same order
dataset=Dataset.load_from_df(ratings_dataset[['userId','movieId','rating']],reader)

#Using the split function to perform cross validation 
dataset.split(n_folds=6)

#Intialising the SVD model and specifying the number of latent features
#we can tune this parameters according to our requirement
svd=SVD(n_factors=25)

#evaluting the model on the based on the root mean square error and Mean absolute error 
evaluate(svd,dataset,measures=['rmse','mae'])

#making the dataset to train our model
train=dataset.build_full_trainset()
#training our model
svd.train(train)



#Making a new series which have two columns in it 
#Movie name and movie id 
movies_dataset = movies_dataset.reset_index()
titles = movies_dataset['movie_name']
indices = pd.Series(movies_dataset.index, index=movies_dataset['movie_name'])
#Function to make recommendation to the user
def recommendataion(user_id,movie):
    result=[]
    #Getting the id of the movie for which the user want recommendation
    ind=indices[movie].iloc[0]
    #Getting all the similar cosine score for that movie
    sim_scores=list(enumerate(cosine_sim[ind]))
    #Sorting the list obtained
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)    
    #Getting all the id of the movies that are related to the movie Entered by the user
    movie_id=[i[0] for i in sim_scores]    
    print('The Movie You Should Watched Next Are --')
    print('ID ,   Name ,  Average Ratings , Predicted Rating ')
    #Varible to print only top 10 movies
    count=0
    for id in range(0,len(movie_id)):
      #to ensure that the movie entered by the user is doesnot come in his/her recommendation
        if(ind != movie_id[id]):
            ratings=ratings_dataset[ratings_dataset['movieId']==movie_id[id]]['rating']
            avg_ratings=round(np.mean(ratings),2)
            #For getting all the movies that a particular user has rated
            rated_movies=list(ratings_dataset[ratings_dataset['userId']==user_id]['movieId'])
            #to take only thoese movies that a particular user and not watched yet
            if(id not in rated_movies):
                #To print only thoese movies which have an average ratings that is more than 3.5
                if(avg_ratings >3.5):
                    count+=1
                    #Getting the movie_id of the corresponding movie_name
                    id_movies=movies_dataset[movies_dataset['movie_name']==titles[movie_id[id]]]['movieId'].iloc[0]
                    predicted_ratings=round(svd.predict(user_id,movie_id[id]).est,2)
                    print(f'{movie_id[id]} , {titles[movie_id[id]]} ,{avg_ratings}, {predicted_ratings}')
                    result.append([titles[movie_id[id]],str('Predicted Rating'),str(predicted_ratings)])
                if(count >=10):
                        break
    return result

#Converting the speech to text using google text to speech api
def speech_to_text(): 
    
    text=''
    sample_rate = 48000
    
    chunk_size = 2048
    #Initialize the recognizer 
    r = sr.Recognizer() 
       
    with sr.Microphone(sample_rate = sample_rate,  
                            chunk_size = chunk_size) as source: 
        #wait for a second to let the recognizer adjust the  
        #energy threshold based on the surrounding noise level 
        r.adjust_for_ambient_noise(source) 
        print ("Speak the name of the movie ")
        #listens for the user's input 
        audio = r.listen(source) 
              
        try: 
            text = r.recognize_google(audio) 
    
            print(text.title())
          
        #error occurs when google could not understand what was said 
          
        except sr.UnknownValueError: 
            print("Google Speech Recognition could not understand audio") 
          
        except sr.RequestError as e: 
            print("Could not request results from Google  Speech Recognition service; {0}".format(e)) 
            
    return text.title()        




#Converting the text to speech using google text to speech api
def text_to_speech():
    file = open('dd.txt','w')
    file.writelines(f'hello User \n')
    file.writelines('The Movie You Should Watched Next and their predicted rating are as follows.  \n') 
    for res in result:
        res=' '.join(res)
        file.write(res+'\n')
        
    file.close()
    file = open('dd.txt','r')
    data= file.read()
    file.close()
    
    
    # Language in which you want to convert 
    language = 'en'
      
    # Passing the text and language to the engine,  
    # here we have marked slow=False. Which tells  
    # the module that the converted audio should  
    # have a high speed 
    myobj = gtts.gTTS(text=data, lang=language, slow=False) 
      
    # Saving the converted audio in a mp3 file named   
    i=random.randint(1,100)
    file='new'+str(i)+'.mp3'
    myobj.save(file) 
      
    #Playing the converted file 
    playsound(file)
    

    
#Getting the output   
i_d=int(input('Enter Your id - '))
movie_name=speech_to_text()
result=recommendataion(i_d,movie_name)
text_to_speech()






    

    
        
