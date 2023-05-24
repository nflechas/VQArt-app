import os
from tqdm import tqdm
from whoosh.index import * # whoosh: full-text indexing and searching
from whoosh.fields import *
from whoosh import qparser
import sys
sys.path.insert(1, "src/data")
#import src.data.wiki_scrape as wiki_scrape
import wiki_scrape as wiki_scrape

class IR(object):
    def __init__(self, 
                 max_passage_length = 800,
                 overlap = 0.4,
                 passages_limit = 10000,
                 data_path = 'data/wiki_articles',
                 index_path = 'index'):
        self.max_passage_length = max_passage_length
        self.overlap = overlap
        self.passages_limit = passages_limit
        self.data_path = data_path
        self.index_path = index_path
        self.ix = None

        passages = self.__load_passages()

        if os.path.exists(self.index_path):
            print(f'Index already exists in the directory {self.index_path}')
            print('Skipping building the index...')
            self.ix = open_dir(self.index_path)
        else:
          self.__create_index(passages)

    def __create_passages_from_article(self, content):
        passages = []
        passage_diff = int(self.max_passage_length * (1-self.overlap))

        for i in range(0, len(content), passage_diff):
            passages.append(content[i: i + self.max_passage_length])
        return passages

    def __scrape_wiki_if_not_exists(self):
        if len(os.listdir(self.data_path)) == 0:
            print('No Wiki articles. Scraping...')
            wiki_scrape.scrape('src/data/entities.txt', 'data/wiki_articles')

    def __load_passages(self):
        self.__scrape_wiki_if_not_exists()

        passages = []
        count = 0
      
        directory = os.fsencode(self.data_path)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if not filename.endswith(".txt"):
               continue

            with open(os.path.join(self.data_path, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                article_passages = self.__create_passages_from_article(content)
                #print(f'Created {len(article_passages)} passages')
                passages.extend(article_passages)

            count += 1
            if count == self.passages_limit:
                break
        return passages

    def __create_index(self, passages):
        # Create the index directory
        os.mkdir(self.index_path)

        # Schema definition:
        # - id: type ID, unique, stored; doc id in order given the passages file
        # - text: type TEXT processed by StemmingAnalyzer; not stored; content of the passage
        schema = Schema(id = ID(stored=True,unique=True),
                        text = TEXT(analyzer=analysis.StemmingAnalyzer())
                        )
    
        # Create an index
        self.ix = create_in("index", schema)
        writer = self.ix.writer() #run once! or restart runtime

        # Add papers to the index, iterating through each row in the metadata dataframe
        id = 0
        for passage_text in tqdm(passages, desc='Building index'): 
            writer.add_document(id=str(id),text=passage_text)
            id += 1
        
        # Save the added documents
        writer.commit()
        print("Index successfully created")

    def retrieve_documents(self, query, topk):
        scores=[]
        text=[]
        passages = self.__load_passages()
        # Open the searcher for reading the index. The default BM25 algorithm will be used for scoring
        with self.ix.searcher() as searcher:
            searcher = self.ix.searcher()
            
            # Define the query parser ('text' will be the default field to search), and set the input query
            q = qparser.QueryParser("text", self.ix.schema, group=qparser.OrGroup).parse(query)
        
            # Search using the query q, and get the n_docs documents, sorted with the highest-scoring documents first
            results = searcher.search(q, limit=topk)
            # results is a list of dictionaries where each dictionary is the stored fields of the document
        
        # Iterate over the retrieved documents
        for hit in results:
            scores.append(hit.score)
            text.append(passages[int(hit['id'])])
        return text, scores
