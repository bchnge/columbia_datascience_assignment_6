from nltk.corpus import movie_reviews
import nltk
import random

from homework_06.src.tagger import document_features
import homework_06.src.classifier as cl
import homework_06.src.tagger as tg

def main():
    #get the movie data
    movie_data = getMovieData()
    #shuffle it up
    random.shuffle(movie_data)
    #break up into test and train
    test_data, train_data = movie_data[:len(movie_data)/2],  movie_data[len(movie_data)/2:]


    #####From here on it's up to you how you want to tag and classify the reviews
    
    # First, determine set of words to use for features. Consider using getTopwords
    
   
    pos_reviews = [sent[0] for sent in train_data if sent[1]=='pos']
    neg_reviews = [sent[0] for sent in train_data if sent[1]=='neg']
    
    # Take out all adjectives
    tagged_pos = tg.posTagger(pos_reviews[0:25], pos_type = 'JJ')
    tagged_neg = tg.posTagger(neg_reviews[0:25], pos_type = 'JJ')
    
    pos_adj = [tag[0] for tag in tagged_pos]
    neg_adj = [tag[0] for tag in tagged_neg]
    
    pos_adj = tg.getTopWords(pos_adj,0.3)
    neg_adj = tg.getTopWords(neg_adj,0.3)
    
    tagged_pos_final = [tag for tag in tagged_pos if tag[0] in pos_adj]
    tagged_neg_final = [tag for tag in tagged_neg if tag[0] in neg_adj]
    
    tagged_features = list(set(tagged_pos_final).union(set(tagged_neg_final))) 
    tagged_features = list(set(tagged_features))
    
    docData = []
    for review in train_data:
        review_features = document_features(review[0], tagged_features)
        docData.append((review[1],review_features))
        
    # must pass docData into the classifier to train.
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
