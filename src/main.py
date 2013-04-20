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
    
    # First, determine set of words to use for features. Consider using getTopwords
    # from a set of reviews that are negative and a set of reviews that are positive
    
    # once we have a list of words, we can use a tagger (either pos or bigram) to get a
    # list of tuples [(word1, tag1),...]. Not sure how useful this really is because
    # document_features doesn't even use the tag portion of the tuple.
    
    # For each review in the training data, use document_features and the list of tagged words
    # to get a set of features 
    
    # Now use that set of features and label to train data
    
    # use the same set of features to test test_data
    
    
    # is there a function already builtin to classifier that calculates the accuracy/classification error?
    
    
 
    
    
    
   
   
def getMovieData():
    """
    Retrieves the movie review data from nltk.corpus and returns a list of tuples of the form (words_in_review, sentiment)
    """
    movie_data = [(movie_reviews.raw(ID), category) for category in movie_reviews.categories() for ID in movie_reviews.fileids(category)]
    return movie_data
    



if __name__ == '__main__':
    movie_data = getMovieData()
