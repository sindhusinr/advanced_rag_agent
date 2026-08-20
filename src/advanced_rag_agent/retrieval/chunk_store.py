import pickle


def load_chunks():

    with open(
        "data/chunks.pkl",
        "rb",
    ) as file:

        chunks = pickle.load(file)

    return chunks