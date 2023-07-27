from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


# LLM model
model = SentenceTransformer('intfloat/e5-small-v2')


def cosine_similarity(a, b):
    return a.dot(b)


# Run this once.
def save_embeddings():
    """Save the embeddings of the hadiths in the dataframe to a csv file."""

    # Read 'source_hadith.csv' into a dataframe
    print('Reading source_hadiths.csv...')
    df = pd.read_csv('source_hadiths.csv')

    df = df[['hadith_text', 'grade', 'takhrij', 'explanation']]
    df['hadith_whole'] = 'query: \n' + \
                    'Book: ' + df['takhrij'] + '\n'\
                    'Grade: ' + df['grade'] + '\n'\
                    'Hadith Text: ' + df['hadith_text']

    print('Start embedding...')
    df['embedding'] = df['hadith_whole'].head(10).apply(lambda x: model.encode(x, normalize_embeddings=True)) \
                                        .head(10).apply(lambda x: tuple(x))
    print(type(df['embedding'][0]))

    # Save the dataframe to a csv file
    print('Saving dataframe to hadith_embeddings.csv...')
    df.to_csv('hadith_embeddings2.csv')
    print('Done saving dataframe to hadith_embeddings.csv')


def find_similar(query: str, top_n: int = 5):
    """Find the top n similar hadiths to the query hadith."""

    # Read 'hadith_embeddings.csv' into a dataframe
    print('Reading hadith_embeddings.csv...')
    df = pd.read_csv('hadith_embeddings2.csv')
    df['embedding'] = df['embedding'].head(10).apply(lambda x: np.array(eval(x)))

    # Embed the query hadith
    print('Embedding query hadith...')
    query_embedding = model.encode(query, normalize_embeddings=True)

    # Calculate the cosine similarity between the query hadith and all the hadiths in the dataframe
    print('Calculating cosine similarity...')    
    df['similarity'] =  df['embedding'].head(10).apply(lambda x: cosine_similarity(x, query_embedding))
    
    # Sort the dataframe by similarity in descending order
    print('Sorting dataframe by similarity...')
    df.sort_values(by=['similarity'], inplace=True, ascending=False)

    # Print the top n similar hadiths
    print(f'Printing top {top_n} similar hadiths...')
    for i in range(top_n):
        print(f'Hadith {i+1}:')
        print(df.iloc[i]['hadith_whole'])
        print(f'Similarity: {format(df.iloc[i]["similarity"])}')
        print()


if __name__ == '__main__':
    save_embeddings()
    
    book: str = '*'
    grade: str = '*'
    hadith_text: str = "aishah reported: ... jihad"

    query: str = 'query: \n' + \
                    'Book: {book}\n' + \
                    'Grade: {grade}\n' + \
                    'Hadith Text: {hadith_text}'
    find_similar(query, 5)

