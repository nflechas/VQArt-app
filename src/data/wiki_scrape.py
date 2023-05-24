import wikipedia
import os

def get_raw_wikipedia_article(entity):
  try:
    results = wikipedia.search(entity)
    best_result = results[0]
    page = wikipedia.page(best_result, auto_suggest=False)
    return page.content
  except wikipedia.exceptions.DisambiguationError as e:
    # Search term can't be disambiguated so we try 
    # again with a more specific search term adding ' (arts)'
    return get_raw_wikipedia_article(entity + ' (arts)')
  
  except wikipedia.exceptions.PageError as e:
    # If the page doesn't exist, handle the PageError here.
    print("The requested page does not exist on Wikipedia.")
    return None

def clean_article(raw_article):
  lines = raw_article.split('\n')
  clean_lines = []
  for l in lines:
    if l.startswith('== See also'):
      break
    if l.startswith('== References'):
      break
    if l.startswith('='):
      continue
    if len(l.strip()) == 0:
      continue
    
    clean_lines.append(l.strip())
  return '\n'.join(clean_lines)

def save_article(content, path):
  with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

def load_entities(entities_path):
  with open(entities_path, 'r', encoding='utf-8') as f:
    return [l.strip() for l in f.readlines()]

def scrape(entities_path, save_path):
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  entities = load_entities(entities_path)
  for entity in entities:
    raw_article = get_raw_wikipedia_article(entity)
    if raw_article == None:
      print(f'Article on Wikipedia not found for entity {entity} :(')
      continue
    
    cleaned_article = clean_article(raw_article)
    save_article(cleaned_article, os.path.join(save_path, f'{entity}.txt'))

if __name__ == '__main__':
  scrape('src/data/entities.txt', 'data/wiki_articles')