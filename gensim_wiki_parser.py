from gensim.corpora.wikicorpus import WikiCorpus

wiki = WikiCorpus('./data/enwiki-latest-pages-articles.xml.bz2')
wiki.saveAsText('./data/wiki_pt_vocab200k')
