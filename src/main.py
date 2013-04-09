from nltk.corpus import movie_reviews
import nltk
import random

from naivebayes.src.tagger import document_features
import naivebayes.src.classifier as cl


def main():
    #get the movie data
    movie_data = getMovieData()
    #shuffle it up
    random.shuffle(movie_data)
    #break up into test and train
    test_data, train_data = movie_data[:len(movie_data)/2],  movie_data[len(movie_data)/2:]


    #####From here on it's up to you how you want to tag and classify the reviews

 
    
    
    
   
   
def getMovieData():
    """
    Retrieves the movie review data from nltk.corpus and returns a list of tuples of the form (words_in_review, sentiment)
    """
    movie_data = [(movie_reviews.raw(ID), category) for category in movie_reviews.categories() for ID in movie_reviews.fileids(category)]
    return movie_data
    



if __name__ == '__main__':
    movie_data = getMovieData()
