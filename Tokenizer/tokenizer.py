import gzip,re,sys
fname="NunavutHansard.en.gz"
#pattern=r'(\$?[^\s][a-zA-Z0-9\'_-]+ | [^a-zA-Z0-9])'
pattern=r'\$?[\w\-\']+|[^\s*S+\W*S]|[^\w\s]+'
with gzip.open(fname, mode='rt',encoding='utf-8') as infile:
    for line in infile:
        for word in line.split(" "):
            if(re.search(pattern,word)):
                #print(word)
                matches=re.finditer(pattern,word)
                for im in matches:
                    print(im.group(0))
              
                   
                            
                         
           
        
           
