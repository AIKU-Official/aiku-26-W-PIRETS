import re

def normalize_text(text):
    """
    [Common Preprocessing Module]
    gen_corpus.py와 gen_qrels.py에서 공통으로 import하여 사용합니다.
    """
    if not text:
        return ""

    text = text.lower()
    text = text.replace("’", "'")
    
    # 1. 메타데이터 태그 삭제
    text = text.replace('', '')
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    
    # 2. '&' -> 'and'
    text = text.replace('&', ' and ')
    
    # 3. 검열 문자(*) 삭제 (Join)
    # f**k -> fk (자음 보존 및 단어 연결)
    text = text.replace('*', '')

    # 4. 괄호/따옴표 등 제거
    text = re.sub(r'[(){}\[\]"]', '', text)

    # 5. Whitelist: 영문, 숫자, 한글, 공백, 작은따옴표만 생존
    # 숫자는 그대로 둠 (Real-World Logic)
    text = re.sub(r"[^a-z0-9가-힣\s']", " ", text)

    # 6. 공백 정리
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text