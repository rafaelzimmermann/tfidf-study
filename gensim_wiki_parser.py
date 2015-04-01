from gensim.corpora.wikicorpus import WikiCorpus
from gensim.corpora import MmCorpus
wiki = WikiCorpus('./data/enwiki-latest-pages-articles.xml.bz2')
MmCorpus.serialize('./data/wiki_pt_vocab200k', wiki)
