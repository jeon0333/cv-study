# CV 깊이 읽기 — 한국어 컴퓨터 비전 학습 사이트

ResNet 논문([Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385))부터 시작해 **왜 → 개념 → 수식 → 구현 → 실험 → 복습**을 따라 학습하는 한국어 사이트입니다.

- **사이트 주소(배포 후):** https://jeon0333.github.io/cv-study/
- **문서:** `docs/`
- **실습 코드:** `cv_models/`
- **GitHub Pages 자동 배포:** `.github/workflows/pages.yml`

> 아직 사이트를 GitHub에 올리지 않았다면 위 URL은 동작하지 않습니다. 먼저 저장소를 만들어 푸시하고 Pages를 활성화해야 합니다.

## 1. GitHub에 첫 공개하기

1. GitHub에서 **Public** 저장소를 만들고 이름을 `cv-study`로 지정합니다. (초기 README / .gitignore 없이 생성해도 됩니다.)
2. 이 프로젝트 폴더에서 다음을 실행합니다. 다른 이름의 저장소를 택했다면 `mkdocs.yml`의 `site_url`, `repo_url`, `repo_name`도 맞춰 바꿔 주세요.

```bash
git init
git add .
git commit -m "docs: start Korean CV learning site"
git branch -M main
git remote add origin https://github.com/jeon0333/cv-study.git
git push -u origin main
```

3. 저장소의 **Settings → Pages → Build and deployment → Source**에서 **GitHub Actions**를 선택합니다.
4. **Actions** 탭의 `Deploy Korean CV Study Site` 작업이 성공하면 GitHub Pages 주소로 접근합니다. Actions가 조직/저장소 정책 때문에 비활성화된 경우 활성화해야 합니다.

> Public 저장소의 GitHub Pages는 무료 플랜에서도 사용할 수 있습니다. 공개를 원하지 않는 경우 GitHub 계정 플랜 및 Pages 정책을 확인하세요.

## 2. 내 컴퓨터에서 미리보기

Python 3.10+ 환경:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-site.txt
mkdocs serve
```

브라우저에서 http://127.0.0.1:8000/ 으로 접속합니다. Markdown 파일을 고치면 미리보기가 자동 갱신됩니다.

```bash
mkdocs build --strict  # 배포 전 검증용
```

## 3. 학습 자료 추가하기

- 기존 단원 수정: `docs/papers/resnet/` 아래의 Markdown 수정
- 다음 논문 추가: `docs/papers/` 아래 새 폴더를 만들고 `docs/templates/add-a-paper.md`를 참고
- 상단/사이드 메뉴 추가: `mkdocs.yml`의 `nav` 수정
- 학습 실험 코드: `cv_models/` 폴더에서 관리
- 학습 진도: `docs/progress.md`의 체크박스는 해당 브라우저에 저장됩니다. GitHub 계정 간 동기화되지 않습니다.

## 4. ResNet 실습 코드 확인

PyTorch를 공식 가이드(https://pytorch.org/get-started/locally/)에 따라 설치한 다음:

```bash
python -m unittest discover -s tests -v
python cv_models/train_cifar10.py --epochs 2 --subset-size 5000
```

첫 명령은 데이터셋 없이 네트워크 출력 형태와 역전파를 검증합니다. 두 번째 명령은 **CIFAR-10 다운로드가 필요**하고, 소규모 교육용 실습이므로 논문에 보고된 ImageNet 수치와 비교하면 안 됩니다.

## 편집 원칙

- **친절한 한국어**로 서술하고, 영어 용어는 첫 등장 때 뜻을 정의합니다.
- 각 단원에 학습 목표, 직관, 예제, 수식, 코드, 오해하기 쉬운 부분, 확인 문제를 넣습니다.
- 원문 주장과 저자의 해석, 자체 실험 결과를 구분합니다.
- **아직 실행하지 않은 실험 수치는 절대 적지 않습니다.**
- 타인의 그림 및 원문 본문을 무단 복제하지 않고 논문 링크와 출처를 남깁니다.

## 참고

- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016: https://arxiv.org/abs/1512.03385
- MkDocs Material: https://squidfunk.github.io/mkdocs-material/
- GitHub Pages custom workflows: https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages

※ 이 프로젝트는 MkDocs Material 9.x를 사용합니다. 유지보수 상황에 따라 후속 문서 도구로 옮길 수 있도록 본문은 표준 Markdown 중심으로 작성했습니다.
