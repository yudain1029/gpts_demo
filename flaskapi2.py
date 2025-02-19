from flask import Flask, request, jsonify
import openai
from pinecone import Pinecone

# 설정: 본인의 API 키, 인덱스 이름, namespace를 입력하세요.
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "pcsk_7WRJZS_FLaUX5GAgGHKVoejiATYhVkL5cUBVExFUNQT4g8hogduXsqLob2S8DNwrNZNSW"
INDEX_NAME = "tox-demo2"
NAMESPACE = "ns3"  # Pinecone에 데이터를 업로드할 때 사용한 namespace
MODEL = "text-embedding-ada-002"

# OpenAI 초기화
openai.api_key = OPENAI_API_KEY

# Pinecone 인스턴스 생성 (최신 방식)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

app = Flask(__name__)

def get_embedding(text):
    response = openai.Embedding.create(input=text, model=MODEL)
    return response["data"][0]["embedding"]

@app.route('/query', methods=['GET'])
def query_paragraph():
    keyword = request.args.get('keyword')
    if not keyword:
        return jsonify({"error": "No keyword provided"}), 400

    # 쿼리 텍스트의 임베딩 생성
    query_vector = get_embedding(keyword)

    # Pinecone 인덱스에서 namespace "ns3" 내에서 검색 (top_k=5)
    results = index.query(
        vector=query_vector,
        top_k=5,
        include_metadata=True,
        namespace=NAMESPACE
    )

    paragraphs = []
    if results and results.get("matches"):
        for match in results["matches"]:
            # 메타데이터에서 "paragraph" 필드를 추출합니다.
            metadata = match.get("metadata", {})
            paragraph = metadata.get("paragraph")
            if paragraph:
                paragraphs.append(paragraph)
        if not paragraphs:
            paragraphs.append("Matches found, but no paragraph metadata available.")
    else:
        paragraphs.append("No relevant information found.")

    return jsonify({"response": paragraphs})

if __name__ == '__main__':
    app.run(debug=True)
