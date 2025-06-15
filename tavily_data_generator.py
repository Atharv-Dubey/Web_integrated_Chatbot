
from tavily import TavilyClient


def tavily_answer_generator(query):

    api_key = "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"

    client = TavilyClient(api_key=api_key)


    try:
        answer = client.qna_search(query=query)
        return answer
    except Exception as e:
        return 0

