# 학습
from src.common import SEED, load_data # 데이터 불러오기 모듈

# 데이터 정제 및 분리 후
# 해당 모듈에서 모델 저장 후 평가 및 고도화 시 모델 블러와서 사용 
# 저장 경로 models\analysis2
# 예시
def test4():
    df = load_data()
    return df.shape

# 해당 페이지를 직접 실행 후 모델 저장
if __name__ == "__main__":
    test4()