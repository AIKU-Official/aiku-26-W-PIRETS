import re
import logging
from jamo import h2j, j2hcj

# 라이브러리 체크
try:
    from g2pk import G2p as G2pKr
    from g2p_en import G2p as G2pEn
    _HAS_LIBRARIES = True
except ImportError:
    _HAS_LIBRARIES = False

logger = logging.getLogger(__name__)

# =========================================================
#  Phoneme Confusion Maps (IPA Version)
# =========================================================

# 1. 한국어 (Korean) - 그대로 유지 (이미 IPA 호환됨)
KOREAN_CONFUSION_MAP = {
    # -----------------------------------------------------
    # 1. 자음 (Consonants) - 기존 유지
    # -----------------------------------------------------
    # 연구개음 (Velar)
    'k': ['kʰ', 'k͈', 'g'],  'g': ['k', 'kʰ', 'k͈'],
    'kʰ': ['k', 'k͈', 'g'],  'k͈': ['k', 'kʰ', 'g'],
    
    # 치조음 (Al104///8/veolar)
    't': ['tʰ', 't͈', 'd'],  'd': ['t', 'tʰ', 't͈'],
    'tʰ': ['t', 't͈', 'd'],  't͈': ['t', 'tʰ', 'd'],
    
    # 양순음 (Bilabial)
    'p': ['pʰ', 'p͈', 'b'],  'b': ['p', 'pʰ', 'p͈'],
    'pʰ': ['p', 'p͈', 'b'],  'p͈': ['p', 'pʰ', 'b'],
    
    # 치찰음 (Sibilant)
    's': ['s͈'], 's͈': ['s'],
    'tɕ': ['tɕʰ', 'tɕ͈', 'dʒ'], 'dʒ': ['tɕ', 'tɕʰ', 'tɕ͈'], # ㅈ
    'tɕʰ': ['tɕ', 'tɕ͈', 'dʒ'], 'tɕ͈': ['tɕ', 'tɕʰ', 'dʒ'], # ㅊ, ㅉ
    
    # 비음/유음 (Nasal/Liquid)
    'n': ['m', 'ŋ'], 'm': ['n', 'ŋ'], 'ŋ': ['n', 'm'],
    'l': ['r'], 'r': ['l'],

    # -----------------------------------------------------
    # 2. 모음 (Vowels) -  [NEW] 현실적 모음 노이즈 추가
    # -----------------------------------------------------
    
    # (1) 'ㅐ(ɛ)' <-> 'ㅔ(e)' : 현대 한국어의 가장 흔한 붕괴
    'ɛ': ['e'], 
    'e': ['ɛ'],
    
    # (2) 'ㅒ(jɛ)' <-> 'ㅖ(je)' : 위와 동일
    'jɛ': ['je'], 
    'je': ['jɛ'],

    # (3) 'ㅓ(ʌ)' <-> 'ㅗ(o)' : "너를" -> "노를" (음향적으로 가까움)
    #     'ㅓ(ʌ)' <-> 'ㅕ(jʌ)' : 활음 첨가/탈락
    'ʌ': ['o', 'jʌ'], 
    'o': ['ʌ', 'jo', 'u'], # ㅗ는 ㅜ와도 가끔 헷갈림 ("나도" -> "나두")
    
    # (4) 'ㅏ(a)' <-> 'ㅑ(ja)' : "아냐" -> "아나"
    'a': ['ja'], 
    'ja': ['a'],

    # (5) 'ㅜ(u)' <-> 'ㅠ(ju)' <-> 'ㅡ(ɯ)'
    'u': ['ju', 'ɯ', 'o'], 
    'ju': ['u'],
    'ɯ': ['u'], # ㅡ <-> ㅜ

    # (6) 'ㅚ(ø)', 'ㅙ(wɛ)', 'ㅞ(we)' : 전부 [we]로 소리남
    # g2p.py의 IPA 매핑: ㅚ->ø, ㅙ->wɛ, ㅞ->we
    'ø': ['wɛ', 'we'],
    'wɛ': ['ø', 'we'],
    'we': ['ø', 'wɛ'],
    
    # (7) 'ㅣ(i)' <-> 'ㅟ(y)' (입술 모양 차이일 뿐 혀 위치 비슷)
    'i': ['y'],
    'y': ['i']
}

# 2. 영어 (English) -  [수정됨] ARPABET -> IPA로 변경
# g2p.py의 ARPABET_TO_IPA 매핑 테이블을 참고하여 키를 맞춤
ENGLISH_CONFUSION_MAP = {
    # Vowels (모음 혼동)
    'ɑ': ['ʌ', 'ɔ'],    # AA <-> AH, AO
    'ʌ': ['ɑ', 'ə'],    # AH <-> AA, AX(schwa)
    'i': ['ɪ'],         # IY <-> IH
    'ɪ': ['i'],         # IH <-> IY
    'u': ['ʊ'],         # UW <-> UH
    'ʊ': ['u'],         # UH <-> UW
    'ɛ': ['æ'],         # EH <-> AE
    'æ': ['ɛ'],         # AE <-> EH
    
    # Plosives (파열음: 유성 <-> 무성)
    'p': ['b'], 'b': ['p'], 
    't': ['d'], 'd': ['t'], 
    'k': ['g'], 'g': ['k'],
    
    # Fricatives (마찰음)
    'f': ['v', 'θ'],    # F <-> V, TH
    'v': ['f', 'ð'],    # V <-> F, DH
    's': ['z', 'ʃ'],    # S <-> Z, SH
    'z': ['s', 'ʒ'],    # Z <-> S, ZH
    'ʃ': ['s', 'ʒ'],    # SH <-> S, ZH
    'ʒ': ['z', 'ʃ'],    # ZH <-> Z, SH
    'θ': ['ð', 'f'],    # TH <-> DH, F (Th-fronting)
    'ð': ['θ', 'v'],    # DH <-> TH, V
    
    # Nasals (비음)
    'm': ['n'], 'n': ['m', 'ŋ'], 'ŋ': ['n']
}


# =========================================================
#  Mapping Tables (To Unified IPA)
# =========================================================

# 1. ARPABET (영어) -> IPA 변환 테이블
# XPhoneBERT가 이해하는 IPA 심볼로 매핑
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ',
    'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'g',
    'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'P': 'p', 'R': 'r', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}

# 2. Hangul Jamo (한국어) -> IPA 변환 테이블
# XPhoneBERT의 학습 데이터와 최대한 유사한 매핑 사용
JAMO_TO_IPA = {
    # 자음 (Initial & Final)
    'ㄱ': 'k', 'ㄲ': 'k͈', 'ㄴ': 'n', 'ㄷ': 't', 'ㄸ': 't͈',
    'ㄹ': 'l', 'ㅁ': 'm', 'ㅂ': 'p', 'ㅃ': 'p͈', 'ㅅ': 's',
    'ㅆ': 's͈', 'ㅇ': 'ŋ', 'ㅈ': 'tɕ', 'ㅉ': 'tɕ͈', 'ㅊ': 'tɕʰ',
    'ㅋ': 'kʰ', 'ㅌ': 'tʰ', 'ㅍ': 'pʰ', 'ㅎ': 'h',
    # 모음 (Vowel)
    'ㅏ': 'a', 'ㅐ': 'ɛ', 'ㅑ': 'ja', 'ㅒ': 'jɛ', 'ㅓ': 'ʌ',
    'ㅔ': 'e', 'ㅕ': 'jʌ', 'ㅖ': 'je', 'ㅗ': 'o', 'ㅘ': 'wa',
    'ㅙ': 'wɛ', 'ㅚ': 'ø', 'ㅛ': 'jo', 'ㅜ': 'u', 'ㅝ': 'wʌ',
    'ㅞ': 'we', 'ㅟ': 'y', 'ㅠ': 'ju', 'ㅡ': 'ɯ', 'ㅢ': 'ɰi',
    'ㅣ': 'i',
    # 종성 예외 처리 (받침 ㅇ은 ŋ, 나머지는 초성과 동일 매핑 사용하되 대표음화는 g2pk가 처리함)
    'ㄳ': 'k', 'ㄵ': 'n', 'ㄶ': 'n', 'ㄺ': 'k', 'ㄻ': 'm',
    'ㄼ': 'p', 'ㄽ': 'l', 'ㄾ': 'l', 'ㄿ': 'p', 'ㅀ': 'l',
    'ㅄ': 'p'
}

class G2PConverter:
    def __init__(self):
        #  __init__에서는 절대 모델을 로드하지 않습니다! (메모리/충돌 방지)
        self.g2p_kr = None
        self.g2p_en = None
        
    def _load_models(self):
        """실제 사용할 때 모델 로드 (Lazy Loading)"""
        # 이미 로드되어 있으면 패스
        if self.g2p_kr is not None:
            return

        if _HAS_LIBRARIES:
            #  중요: worker 프로세스 내에서는 반드시 CPU 모드로 강제해야 충돌 안 남
            # g2pk는 device 인자가 없으면 자동으로 cuda를 잡으려다 죽을 수 있음
            # (g2pk 구현체에 따라 다르지만, 안전하게 CPU로 돌리는 게 정신건강에 좋음)
            try:
                self.g2p_kr = G2pKr() # g2pk 내부적으로 CPU 사용하도록 유도 필요할 수 있음
                self.g2p_en = G2pEn()
            except Exception as e:
                logger.error(f"Failed to load G2P models: {e}")
        
    def is_hangul(self, char):
        return '\uac00' <= char <= '\ud7a3'

    def _convert_kor_to_ipa(self, text):
        # 1. 발음 변환 (국물 -> 궁물)
        pronounced = self.g2p_kr(text)
        # 2. 자모 분리 (궁물 -> ㄱ ㅜ ㅇ ㅁ ㅜ ㄹ)
        jamo_str = j2hcj(h2j(pronounced))
        
        ipa_list = []
        for char in jamo_str:
            if char in JAMO_TO_IPA:
                ipa_list.append(JAMO_TO_IPA[char])
            elif char.strip(): # 공백이 아니면 그냥 둠
                ipa_list.append(char)
        return ipa_list

    def _convert_eng_to_ipa(self, text):
        # g2p_en returns ['AH0', 'L', 'AH1', 'V']
        arpabet_tokens = self.g2p_en(text)
        ipa_list = []
        
        for token in arpabet_tokens:
            # 숫자(강세) 제거 (AH0 -> AH)
            core_token = re.sub(r'\d+', '', token)
            if core_token in ARPABET_TO_IPA:
                ipa_list.append(ARPABET_TO_IPA[core_token])
            elif token.strip():
                ipa_list.append(token) # 특수문자 등
        return ipa_list

    def __call__(self, text):
        """
        Input: "I love you 사랑해"
        Output: "aɪ l ʌ v j u s a r a ŋ h ɛ" (Unified IPA)
        """
        if not text or not _HAS_LIBRARIES:
            return text
        
        #  사용하기 직전에 로드 (각 프로세스별로 독립적으로 로드됨)
        self._load_models()

        # 한영 혼용 처리를 위해 단어(어절) 단위로 쪼개서 판별
        words = text.split()
        final_phonemes = []

        for word in words:
            if re.search(r'[가-힣]', word):
                # self._convert_kor_to_ipa 내부에서 self.g2p_kr 사용
                final_phonemes.extend(self._convert_kor_to_ipa(word))
            else:
                final_phonemes.extend(self._convert_eng_to_ipa(word))
                
        return " ".join(final_phonemes)