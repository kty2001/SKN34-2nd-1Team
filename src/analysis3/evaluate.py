# 평가
from src.common import SEED, load_data # 데이터 불러오기 모듈

# 저장 한 모델을 불러와서 평가
# 예시
def test3():
    df = load_data()
    return df.columns.tolist()
