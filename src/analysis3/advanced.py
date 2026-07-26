# 고도화
from src.common import load_data # 데이터 불러오기 모듈

# 저장 한 모델을 불러와서 고도화 후 최종적으로 모델 저장
# 저장 경로 models\analysis3
# 예시
def test1():
    df = load_data()
    return df.dtypes.to_dict()

# 해당 페이지를 직접 실행 후 모델 저장
if __name__ == "__main__":
    test1()