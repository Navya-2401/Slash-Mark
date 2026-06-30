import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_documents():
    """Reads all .txt files in the same folder."""
    documents = []
    filenames = []
    for file in os.listdir('.'):
        if file.endswith('.txt') and file != 'requirements.txt':
            with open(file, 'r', encoding='utf-8') as f:
                documents.append(f.read())
                filenames.append(file)
    return documents, filenames

def check_plagiarism():
    # 1. Load the text files
    docs, names = load_documents()
    
    if len(docs) < 2:
        print("Need at least two .txt files to compare!")
        return

    # 2 & 3. Clean the text and turn it into vectors (Numbers)
    # TfidfVectorizer automatically makes text lowercase and ignores punctuation!
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(docs)

    # 4. Compare the vectors to find the similarity score
    similarity_matrix = cosine_similarity(vectors)

    # Print the results in the format requested by your project
    print("--- Plagiarism Report ---")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = similarity_matrix[i][j]
            # Print the two file names and their similarity score
            print(f"('{names[i]}', '{names[j]}', {score})")

# Run the program
if __name__ == '__main__':
    check_plagiarism()