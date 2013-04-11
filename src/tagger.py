from collections import defaultdict
import nltk
import pdb



def document_features(document, tagger_output):
    """
    This function takes a document and a tagger_output=[(word,tag)]
    (see functions below), and tells you which words were present as
    'words' (as opposed to 'tags') in tagger_output.

    Parameters
    ----------
    document: string, your document
    tagger_output: list of tuples of the form [(word, tag)]
    
    Returns
    -------
    features : dictionary of tuples ('has(word)': Boolean)

    Notes
    -----
    Use the nltk.word_tokenize() to break up your text into words 
    """
    doc_words = nltk.word_tokenize(document)
    feature_words = [t[0] for t in tagger_output]
    features = checkFeatures(doc_words, feature_words)

    return features
    

def checkFeatures(document, feature_words):
    """
    This function takes a document and a list of feautures, i.e. words you
    have identitifed as features, and returns a dictionary telling you which
    words are in the document

    Parameters
    ----------
    document: list of strings (words in the text you are analyzing)
    features: list of strings (words in your feature list)

    Returns
    -------
    features: dictionary
        keys are Sting (the words)
        values are Boolean (True if word in feature_words)
    """
    wordsInDoc = set(document)
    features = {word: word in wordsInDoc for word in feature_words}

    return features


def onlyAlpha(document):
    """
    Takes a list of strings in your document and gets rid of everything that
    is not alpha, i.e. returns only words

    Parameters
    ----------
    document: list of strings

    Returns
    -------
    words: list of strings
    """
    words = [word for word in document if word.isalpha()]

    return words


def getTopWords(word_list, percent):
    """
    Takes a word list and returns the top percent of freq. of occurence.
    I.e. if percent = 0.3, then return the top 30% of word_list.
    
    Parameters
    ----------
    word_list: list of words
    percent: float in [0,1]

    Returns
    -------
    top_words: list 

    Notes
    -----
    Make sure this returns only alpha character strings, i.e. just words.
    Also, consider using the nltk.FreqDist()
    """
    ###get rid of non alphas in case you have any
    word_list = onlyAlpha(word_list)
    wordFreqDist = nltk.FreqDist(word_list)
    cutOffIndex = int(len(wordFreqDist)*percent)
    top_words = wordFreqDist.keys()[: cutOffIndex + 1]

    return top_words


def posTagger(documents, pos_type=None):
    """
    Return all unique part of speech tags in documents.

    Takes a list of strings, i.e. your documents, and tags all the words in
    each string using the nltk.pos_tag().
    In addition if pos_type is not None the function will return only tuples
    (word, tag) tuples where tag is of type pos_type. For example, if
    pos_type = 'NN' we will get back all words tagged with "NN" "NNP" "NNS" etc

    Parameters
    ----------
    documents: list of strings
    pos_type: string

    Returns
    -------
    tagged_words: list of tuples (word, pos)
        One single list no matter how many documents you have.  

    Notes
    -----
    You need to turn each string in your documents list into a list of words
    and you want to return a list of unique (word, tag) tuples. Use the 
    nltk.word_tokenize() to break up your text into words but MAKE SURE you
    return only alpha characters words.  The order of the returned list does
    not matter.
    """
    tagged_word_set = set()
    for document in documents:
        doc_words = nltk.word_tokenize(document)
        tags = nltk.pos_tag(doc_words)
        tagged_word_set.update(set(tags))

    if pos_type:
        tagged_words = [
            tag for tag in tagged_word_set if tag[0].isalpha 
            and tag[1].startswith(pos_type)]
    else:
        tagged_words = [tag for tag in tagged_word_set if tag[0].isalpha]

    return tagged_words


def bigramTagger(train_data, docs_to_tag, base_tagger=posTagger, pos_type=None):
    """
    Takes a list of strings, i.e. your documents, trains a bigram tagger using the base_tagger for a first pass, then tags all the words in the documents. In addition if pos_type is not None the function will return only those (word, tag) tuples where tag is of type pos_type. For example, if pos_type = 'NN' we will get back all words tagged with "NN" "NNP" "NNS" etc

    Parameters
    ----------
    train_data: list of tuples (word, tag), for trainging the tagger
    docs_to_tag: list of strings, the documents you want to extract tags from
    pos_type: string

    Returns
    -------
    tagged_words: list of tuples (word, pos)

    Notes
    -----
    You need to turn each string in your documents list into a list of words and you want to return a list of unique (word, tag) tuples. Use the nltk.word_tokenize() to break up your text into words but MAKE SURE you return only alpha characters words. Also, note that nltk.bigramTagger() is touchy and doesn't like [(word,tag)] - you need to make this a list of lists, i.e. [[(word,tag)]]
    """
    pos_tag = base_tagger(train_data)
    bigramTagger = nltk.BigramTagger([pos_tag])

    tagged_words = set()
    for document in docs_to_tag:
        doc_words = nltk.word_tokenize(document)
        tags = bigramTagger.tag(doc_words)
        [tagged_words.add(tag) for tag in tags]
    if pos_type:
        tagged_words = [tag for tag in tagged_words if tag[1] and tag[0].isalpha and tag[1].startswith(pos_type)]
    else:
        tagged_words = [tag for tag in tagged_words if tag[0].isalpha]

    return tagged_words

    
    





    







