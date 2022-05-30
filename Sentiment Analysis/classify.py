import math, re
from queue import Empty
from tkinter.tix import Tree
from turtle import pos


def tokenize(s):
    tokens = s.lower().split()
    trimmed_tokens = []
    for t in tokens:
        if re.search('\w', t):
            # t contains at least 1 alphanumeric character
            t = re.sub('^\W*', '', t) # trim leading non-alphanumeric chars
            t = re.sub('\W*$', '', t) # trim trailing non-alphanumeric chars
        trimmed_tokens.append(t)
    return trimmed_tokens

# A most-frequent class baseline
class Baseline:
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        self.mfc = sorted(klass_freqs, reverse=True, 
                          key=lambda x : klass_freqs[x])[0]
    
    def classify(self, test_instance):
        print (self.mfc)

class Lexicon:
    positive = 0
    negative = 0
    neutral = 0
    def __init__(self, klasses):
        self.train(klasses)

    def train(self, klasses):
        # Count classes to determine which is the most frequent
        klass_freqs = {}
        for k in klasses:
            klass_freqs[k] = klass_freqs.get(k, 0) + 1
        #print(klass_freqs)
        self.mfc = sorted(klass_freqs, reverse=True, 
                          key=lambda x : klass_freqs[x])[0]
        #print(self.mfc)
    
    def classify(self, test_instance):
        #return self.mfc
        pos_count = 0
        neg_count = 0
        neutral_count = 0
        for i in tokenize(test_instance):
            if i in pos_texts:
                pos_count = pos_count + 1
            if i in neg_texts:
                neg_count = neg_count + 1
        if pos_count == neg_count:
            print("neutral")
        elif pos_count > neg_count:
            print("positive")
        elif pos_count < neg_count:
            print("negative")
        

class NB:
    def __init__(self, klasses, texts):
        self.train(klasses, texts)

    def train(self, klasses, texts):
        # Count classes to determine which is the most frequent
        self.klass_freqs = {}
        self.words = {}
        self.pos_words = {}
        self.neg_words = {}
        self.neutral_words = {}
        self.pos_count = 0
        self.neg_count = 0
        self.neutral_count = 0
        self.classes = 0
        j = 0
        for k in klasses:
            self.klass_freqs[k] = self.klass_freqs.get(k, 0) + 1
            self.classes = self.classes + 1

        self.mfc = sorted(self.klass_freqs, reverse=True, 
                          key=lambda x : self.klass_freqs[x])[0]
        for t in texts:
            if klasses[j] == 'positive':
                for i in tokenize(t):
                    self.words[i] = self.words.get(i,0) + 1
                    self.pos_words[i] = self.pos_words.get(i,0) + 1
                    self.pos_count += 1
            elif klasses[j] == 'negative':
                for i in tokenize(t):
                    self.words[i] = self.words.get(i,0) + 1
                    self.neg_words[i] = self.neg_words.get(i,0) + 1 
                    self.neg_count += 1
            else:
                for i in tokenize(t):
                    self.words[i] = self.words.get(i,0) + 1
                    self.neutral_words[i] = self.neutral_words.get(i,0) + 1
                    self.neutral_count += 1
            j += 1
            #self.klass_freqs[k] = self.klass_freqs.get(k, 0) + 1
            
        self.types = len(self.words)


    def classify(self, test_instance):
        pos_prob = math.log(self.klass_freqs["positive"])-math.log(self.classes)
        neg_prob = math.log(self.klass_freqs["negative"])-math.log(self.classes)
        neutral_prob = math.log(self.klass_freqs["neutral"])-math.log(self.classes)
        for i in tokenize(test_instance):
            if ((self.pos_words.get(i) is not None) and (self.neg_words.get(i)is not None) and (self.neutral_words.get(i)is not None)):
                pos_prob = pos_prob + math.log((self.pos_words.get(i) + 1) /(self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((self.neg_words.get(i) + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((self.neutral_words.get(i) + 1) / (self.neutral_count + len(self.words)))
            elif (self.pos_words.get(i) is not None and (self.neg_words.get(i)is not None) and (self.neutral_words.get(i)==None)): 
                pos_prob = pos_prob + math.log((self.pos_words.get(i) + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((self.neg_words.get(i) + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((0 + 1) / (self.neutral_count + len(self.words)))
            elif(self.pos_words.get(i) is not None and (self.neg_words.get(i)==None) and (self.neutral_words.get(i)is not None)):
                pos_prob = pos_prob + math.log((self.pos_words.get(i) + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((self.neutral_words.get(i) + 1) / (self.neutral_count + len(self.words)))

            elif(self.pos_words.get(i) is not None and (self.neg_words.get(i)==None) and (self.neutral_words.get(i)==None)):
                pos_prob = pos_prob + math.log((self.pos_words.get(i) + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((0 + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((0 + 1) / (self.neutral_count + len(self.words)))

            elif(self.pos_words.get(i) is None and (self.neg_words.get(i)is not None) and (self.neutral_words.get(i)is not None)):
                pos_prob = pos_prob + math.log((0 + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((self.neg_words.get(i) + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((self.neutral_words.get(i) + 1) / (self.neutral_count + len(self.words)))
            
            elif(self.pos_words.get(i) is None and (self.neg_words.get(i)is not None) and (self.neutral_words.get(i) is None)):
                pos_prob = pos_prob + math.log((0 + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((self.neg_words.get(i) + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((0 + 1) / (self.neutral_count + len(self.words)))

            elif(self.pos_words.get(i) is None and (self.neg_words.get(i) is None) and (self.neutral_words.get(i) is not None)):
                pos_prob = pos_prob + math.log((0 + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((0 + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((self.neutral_words.get(i) + 1) / (self.neutral_count + len(self.words)))

            elif(self.pos_words.get(i) is  None and (self.neg_words.get(i) is  None) and (self.neutral_words.get(i) is  None)):
                pos_prob = pos_prob + math.log((1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((1) / (self.neutral_count + len(self.words)))
        
        if pos_prob > neg_prob and pos_prob > neutral_prob:
            print("positive")    
        elif neg_prob > pos_prob and neg_prob > neutral_prob:
            print("negative")
        elif neutral_prob > pos_prob and neutral_prob > neg_prob:
            print("neutral")
        elif (pos_prob == neg_prob and pos_prob > neutral_prob):
            print('negative')
        elif (pos_prob == neg_prob and pos_prob < neutral_prob):
            print('neutral')
       # return self.mfc

class NBBIN:
    def __init__(self, klasses, texts):
        self.train(klasses, texts)

    def train(self, klasses, texts):
        # Count classes to determine which is the most frequent
        self.klass_freqs = {}
        self.words = {}
        self.pos_words = {}
        self.neg_words = {}
        self.pos_bin_words = {}
        self.neg_count = 0
        self.pos_count = 0
        self.neutral_count = 0
        self.neg_bin_words = {}
        self.neutral_bin_words = {}
        j = 0
        self.classes = 0
        for k in klasses:
            self.klass_freqs[k] = self.klass_freqs.get(k, 0) + 1
            self.classes += 1
        self.mfc = sorted(self.klass_freqs, reverse=True, 
                          key=lambda x : self.klass_freqs[x])[0]
       # self.classes = len(klasses)
        for t in texts:
            tokens = tokenize(t)
            t = set (tokens)
            if klasses[j] == 'positive':
                for i in t: 
                    self.words[i] = self.words.get(i,0) + 1
                    self.pos_bin_words[i] = self.pos_bin_words.get(i,0) + 1
                    self.pos_count += 1
            elif klasses[j] == 'negative':
                for i in t:
                    self.words[i] = self.words.get(i,0) + 1
                    self.neg_count = self.neg_count + 1
                    self.neg_bin_words[i] = self.neg_bin_words.get(i,0) + 1
            else:
                for i in t:
                    self.words[i] = self.words.get(i,0) + 1
                    self.neutral_bin_words[i] = self.neutral_bin_words.get(i,0) + 1
                    self.neutral_count += 1
            j += 1
            
   
    def classify(self, test_instance):
        pos_prob = math.log(self.klass_freqs["positive"]/self.classes)
        neg_prob = math.log(self.klass_freqs["negative"]/self.classes)
        neutral_prob = math.log(self.klass_freqs["neutral"]/ self.classes)
        tokens = tokenize(test_instance)
        tokens = set (tokens)
        for i in tokens:
            if ((self.pos_bin_words.get(i) is not None) and (self.neg_bin_words.get(i)is not None) and (self.neutral_bin_words.get(i)is not None)):
                pos_prob = pos_prob + math.log((self.pos_bin_words.get(i) + 1) /(self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((self.neg_bin_words.get(i) + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((self.neutral_bin_words.get(i) + 1) / (self.neutral_count + len(self.words)))
            elif (self.pos_bin_words.get(i) is not None and (self.neg_bin_words.get(i)is not None) and (self.neutral_bin_words.get(i)==None)): 
                pos_prob = pos_prob + math.log((self.pos_bin_words.get(i) + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((self.neg_bin_words.get(i) + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((0 + 1) / (self.neutral_count + len(self.words)))
            elif(self.pos_bin_words.get(i) is not None and (self.neg_bin_words.get(i)==None) and (self.neutral_bin_words.get(i)is not None)):
                pos_prob = pos_prob + math.log((self.pos_bin_words.get(i) + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((self.neutral_bin_words.get(i) + 1) / (self.neutral_count + len(self.words)))

            elif(self.pos_bin_words.get(i) is not None and (self.neg_bin_words.get(i)==None) and (self.neutral_bin_words.get(i)==None)):
                pos_prob = pos_prob + math.log((self.pos_bin_words.get(i) + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((1) / (self.neutral_count + len(self.words)))

            elif(self.pos_bin_words.get(i) is None and (self.neg_bin_words.get(i)is not None) and (self.neutral_bin_words.get(i)is not None)):
                pos_prob = pos_prob + math.log((1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((self.neg_bin_words.get(i) + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((self.neutral_bin_words.get(i) + 1) / (self.neutral_count + len(self.words)))
            
            elif(self.pos_bin_words.get(i) is None and (self.neg_bin_words.get(i)is not None) and (self.neutral_bin_words.get(i) is None)):
                pos_prob = pos_prob + math.log(1 / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((self.neg_bin_words.get(i) + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((1) / (self.neutral_count + len(self.words)))

            elif(self.pos_bin_words.get(i) is None and (self.neg_bin_words.get(i) is None) and (self.neutral_bin_words.get(i) is not None)):
                pos_prob = pos_prob + math.log((0 + 1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((0 + 1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((self.neutral_bin_words.get(i) + 1) / (self.neutral_count + len(self.words)))

            elif(self.pos_bin_words.get(i) is  None and (self.neg_bin_words.get(i) is  None) and (self.neutral_bin_words.get(i) is  None)):
                pos_prob = pos_prob + math.log((1) / (self.pos_count + len(self.words)))
                neg_prob = neg_prob + math.log((1) / (self.neg_count + len(self.words)))
                neutral_prob = neutral_prob + math.log((1) / (self.neutral_count + len(self.words)))
                
        if pos_prob > neg_prob and pos_prob > neutral_prob:
            print("positive")    
        elif neg_prob > pos_prob and neg_prob > neutral_prob:
            print("negative")
        elif neutral_prob > pos_prob and neutral_prob > neg_prob:
            print("neutral")
        elif (pos_prob == neg_prob and pos_prob > neutral_prob):
            print("negative")
        elif (pos_prob == neg_prob and pos_prob < neutral_prob):
            print("neutral")
        
        return self.mfc

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # Method will be one of 'baseline', 'lr', 'lexicon', 'nb', or
    # 'nbbin'
   
    method = sys.argv[1]
    train_texts_fname = sys.argv[2]
    train_klasses_fname = sys.argv[3]
    test_texts_fname = sys.argv[4]

    pos_texts = [x.strip() for x in open('pos-words.txt',encoding='utf8')]
    neg_texts = [x.strip() for x in open('neg-words.txt',encoding='utf8')]

    train_texts = [x.strip() for x in open(train_texts_fname,
                                           encoding='utf8')]
    train_klasses = [x.strip() for x in open(train_klasses_fname,
                                             encoding='utf8')]
    test_texts = [x.strip() for x in open(test_texts_fname,
                                          encoding='utf8')]
    
    if method == 'baseline':
        classifier = Baseline(train_klasses)
        results = [classifier.classify(x) for x in test_texts]

    elif method == 'lr':
        # Use sklearn's implementation of logistic regression
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        
        count_vectorizer = CountVectorizer(analyzer=tokenize)

        train_counts = count_vectorizer.fit_transform(train_texts)

        lr = LogisticRegression(multi_class='multinomial',
                                solver='sag',
                                penalty='l2',
                                max_iter=1000,
                                random_state=0)
        clf = lr.fit(train_counts, train_klasses)
        
        test_counts = count_vectorizer.transform(test_texts)
        results = clf.predict(test_counts)

    elif method == 'lexicon':
        classifier = Lexicon(train_klasses)
        results = [classifier.classify(x) for x in test_texts]
        #print("positive",classifier.positive,"negative",classifier.negative,"neutral",classifier.neutral)
        #print(results)

    elif method == 'nb':
        classifier_nb = NB(train_klasses, train_texts)
        results = [classifier_nb.classify(x) for x in test_texts]
    
    elif method == 'nbbin':
        classifier = NBBIN(train_klasses, train_texts)
        results = [classifier.classify(x) for x in test_texts]

