import sys
import importlib

# 필수 라이브러리 목록
required_libraries = ["cv2", "numpy", "pyzbar"]


def check_imports():
    missing_libraries = []
    for lib in required_libraries:
        try:
            importlib.import_module(lib)
            print(f"✅ {lib} 라이브러리 로드 성공")
        except ImportError:
            print(f"❌ {lib} 라이브러리 로드 실패")
            missing_libraries.append(lib)

    if missing_libraries:
        print("\n❌ 다음 라이브러리를 설치해야 합니다:")
        for lib in missing_libraries:
            print(f"  pip install {lib}")


def check_python_version():
    print(f"✅ Python 버전: {sys.version}")


print("🔍 [환경 설정 테스트 시작] 🔍")
check_python_version()
check_imports()
print("\n✅ [환경 설정 테스트 완료]")