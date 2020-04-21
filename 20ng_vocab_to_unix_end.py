# Script for converting line endings in original vocabulary file to Unix line endings
# Run with <py 20ng_vocab_to_unix_end.py>

# 20 Newsgroups dataset directory
ng_dir = "data/20news_clean/"

original = ng_dir + "vocab.pkl"
destination = ng_dir + "vocab_unix.pkl"

content = ''
with open(original, 'rb') as infile:
    content = infile.read()
    
outsize = 0
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))