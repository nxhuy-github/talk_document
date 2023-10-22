from TalkDocument import TalkDocument
from helper.keys import *

if __name__ == "__main__":
    td = TalkDocument(
        data_source_path='data/test.txt',
        HF_API_TOKEN=HF_API_TOKEN
    )

    td.get_document()
    documents_splitted = td.get_split(split_type="token")
    print(type(documents_splitted), len(documents_splitted))
    print(documents_splitted)

    # td.create_db_document()

    # question = "What is Hierarchy 4.0?"
    # relevant_splits = td.do_question(question=question)
    # print("TYPE RESPONSE: ", type(relevant_splits))
    # print("Length: ", len(relevant_splits))
    # print(relevant_splits)