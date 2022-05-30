import io, math, re, sys

# A simple tokenizer. Applies case folding
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

# Find the most similar response in terms of token overlap.
def most_sim_overlap(query, responses):
    q_tokenized = tokenize(query)
    max_sim = 0
    max_resp = "Sorry, I don't understand"
    for r in responses:
        r_tokenized = tokenize(r)
        sim = len(set(r_tokenized).intersection(q_tokenized))
        if sim > max_sim:
            max_sim = sim
            max_resp = r
    return max_resp

def w2v(query, response_vector, word_vectors, denum):
    q_tokenized = tokenize(query)
    size = len(q_tokenized)
    query_vector = {}
    line_vector = []
    query_denum = 0
    response_denum = 0
    max_resp = "Sorry, I don't understand"
    for q in q_tokenized:
        if q in word_vectors:
            line_vector.append(word_vectors[q])
            
    query_vector[query] =  [sum(x)/size for x in zip(*line_vector)]
    cosine = {}
    for r in response_vector:
        if len(response_vector[r])!=0:
            num = 0
            query_square = 0
            response_square = 0
            for i in range(0,len(query_vector[query])):
                num += query_vector[query][i] * response_vector[r][i]
                query_square += query_vector[query][i] * query_vector[query][i]
                response_square += response_vector[r][i] * response_vector[r][i]
            query_denum = math.sqrt(query_square)
            response_denum = math.sqrt(response_square)
            if query_denum == 0 or response_denum == 0:
                return max_resp
            cosine[r] = num/(query_denum*response_denum)
    return str([key for key in cosine if cosine[key] == max(cosine.values())]).strip("['']")
    

# Code for loading the fasttext (word2vec) vectors from here (lightly
# modified): https://fasttext.cc/docs/en/crawl-vectors.html
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, tokens[1:]))
        
    return data

if __name__ == '__main__':
    # Method will be one of 'overlap' or 'w2v'
    method = sys.argv[1]
    responses_fname = 'gutenberg.txt'
    vectors_fname = 'cc.en.300.vec.10k'

    responses = [x.strip() for x in open(responses_fname)]

    # Only do the initialization for the w2v method if we're using it
    if method == 'w2v':
        print("Loading vectors...")
        word_vectors = load_vectors(vectors_fname)
        # Hint: Build your response vectors here
        line_vector = []
        response_vector = {}
        response_word_vectors = {}
        denum = {}
        j = 0
        size = 1
        for r in responses:
            tokens = tokenize(r)
            size = len(tokens)
            for t in tokens:
                if t in word_vectors:
                    square = 0
                    if t not in denum:
                        for i in range(0,len(word_vectors[t])):
                            square += word_vectors[t][i] * word_vectors[t][i]
                        denum[t] = math.sqrt(square)
                        for i in range(0,len(word_vectors[t])):
                            word_vectors[t][i] = word_vectors[t][i]/denum[t]
                    line_vector.append(word_vectors[t])
            response_vector[r] =  [sum(x)/size for x in zip(*line_vector)]
            line_vector = []
            

    print("Hi! Let's chat")
    while True:
        query = input()
        if method == 'overlap':
            response = most_sim_overlap(query, responses)
        elif method == 'w2v':
            response = w2v(query,response_vector, word_vectors, denum)

        print(response)
