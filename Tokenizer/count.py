import gzip,sys,re
fname = sys.argv[1]
freq_token={}
freq_word={}
with open(fname,mode='rt') as infile:
    for line in infile:
        if re.match("\\s+",line):
            continue
        line = line.strip()
        freq_token[line.lower()]=freq_token.get(line.lower(),0)+1
        if line.isalpha():
            freq_word[line.lower()]=freq_word.get(line.lower(),0)+1

#counting frequency
for i in sorted(freq_token, key=lambda x : freq_token[x], reverse=True):
    print(i,freq_token[i])
print()


#most frequent word
#for i in sorted(freq_word, key=lambda x : freq_word[x], reverse=True)[:1]:
#    print(i,freq_word[i])

#counting hapex words
#for i in freq_token.keys():
#    if freq_token[i]==1:
#        count_hapex += 1
