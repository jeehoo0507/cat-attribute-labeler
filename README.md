# Cat Attribute Labeler

외부 고양이 사진을 내려받아 `Qwen/Qwen3-VL-32B-Instruct`로 두 번 관찰하고,
자세·눈·입·혀·이빨·착용물·장소 관계의 라벨 초안을 만든다.
기본 실행 대상은 NVIDIA RTX A5000 24GB 한 장이며 NF4 4bit, batch 1, SDPA를 사용한다.

슈딩이 보유 사진은 입력에 포함하지 않는다. 이 레포는 데이터 수집·주석 도구이며
임베딩 모델 학습은 별도 작업이다. 모든 자동 출력은 `unreviewed`로 저장된다.

## 서버에서 클론부터 실행

Linux x86_64, Python 3.11/3.12, NVIDIA 드라이버 570 이상을 기준으로 한다.
잠금 파일의 torch 2.9.1 Linux 배포판은 CUDA 12.8 런타임을 포함한다.
시스템 CUDA toolkit을 따로 설치하기보다 먼저 드라이버와 `nvidia-smi`를 확인한다.

```bash
git clone https://github.com/jeehoo0507/cat-attribute-labeler.git
cd cat-attribute-labeler
./setup.sh

cp .env.example .env
# .env의 /data 경로를 서버에서 쓰기 가능한 경로로 수정
nano .env
set -a
source .env
set +a

./preflight.sh
./prepare_data.sh
./smoke_test.sh
```

`setup.sh`는 의존성과 CPU 회귀 테스트만 실행한다.
`prepare_data.sh`는 실제 사진을 내려받는다.
`smoke_test.sh`에서 모델 원본 가중치를 내려받고 실제 두 사진의 2단계 추론을 수행한다.
통과 여부는 실행·JSON 형식 검사이며 라벨 정확도는 사진과 대조해서 검수한다.
모델 다운로드만 먼저 진행하려면 `./download_model.sh`를 실행한다.

32B NF4도 부가 메모리 때문에 24GB에서 입력 크기에 따라 OOM이 날 수 있다.
사전 점검은 사용 가능한 VRAM 21GiB 이상을 요구하며 실제 GPU smoke로 실행 여부를 확인한다.
OOM이 발생하면 자동으로 작은 모델로 바꾸지 않는다. 설정을 바꿀 때는 새 OUTPUT_ROOT와 RUN_NAME을 쓴다.
원본 BF16 모델 파일을 받아 로딩 시 양자화하므로 **디스크는 최소 85GiB**, 확장 수집은 150GiB 이상을 권장한다.
4bit는 GPU 상주 메모리를 줄이고 원본 다운로드 크기는 줄이지 않는다.

먼저 50장으로 주석 결과를 확인하려면 다음을 실행한다.

```bash
.venv/bin/python -m cat_labeler.label --config "$CONFIG" --limit 50
./status.sh
```

같은 명령을 다시 실행하면 완료한 사진과 완료한 1차 관찰을 재사용한다.
모델 revision, 프롬프트, 설정, 입력 SHA가 바뀌면 기존 결과와 섞지 않고 중단한다.

## 500장 파일럿 전체 실행 및 결과 푸시

서버에 Git push 권한을 준비한다. HTTPS 사용 시 `gh auth setup-git` 또는 서버의
Git credential 설정을 이용한다. 토큰은 `.env`나 결과 파일에 기록하지 않는다.
Git 작성자 정보가 없다면 이 레포에서 `git config user.name` / `git config user.email`도 설정한다.

```bash
# 위 .env를 source한 셸에서 실행
mkdir -p "$OUTPUT_ROOT"
nohup ./run_server.sh > "$OUTPUT_ROOT/server.log" 2>&1 < /dev/null &
tail -f "$OUTPUT_ROOT/server.log"

# 다른 셸에서는 같은 .env를 먼저 source
./status.sh
```

또는 `tmux new -s cat-labeler` 안에서 `./run_server.sh`를 실행한다.
`tail`에서 Ctrl+C는 로그 보기만 종료한다. 실행을 중단하려면 nohup 작업 PID 또는 tmux 작업에 SIGINT를 보낸다.
다시 같은 명령을 실행하면 남은 항목을 처리한다.

전체 순서는 `preflight → 다운로드 → 실제 GPU smoke → 전체 라벨링 → 결과 export → Git push`다.
단계가 실패하면 다음 단계로 넘어가지 않는다. 결과 푸시는 기본으로 켜져 있고 `PUBLISH_RESULTS=0`으로 끌 수 있다.
푸시에만 실패했다면 다음 명령으로 다시 시도한다.

```bash
./publish_results.sh
```

자동 푸시는 현재 checkout 브랜치에 `results/$RUN_NAME/`만 commit한다.
이미 stage된 변경이 있으면 중단한다. force push와 원격 충돌의 자동 병합은 하지 않는다.
원격이 앞서 있으면 결과 commit을 보존한 상태에서 `git pull --rebase`로 해결하고 다시 푸시한다.

작업한 컴퓨터에서 결과를 가져올 때는 다음을 실행한다.

```bash
git pull --ff-only
ls results/pilot32b_v1
```

## 다운로드 범위

| 설정 | COCO 2017 train의 Cat | Oxford Pet의 고양이 | Open Images Cat | 요청 합계 |
| --- | ---: | ---: | ---: | ---: |
| `configs/pilot.json` | 250 | 150 | validation에서 100 | 500 |
| `configs/scale.json` | 최대 5,000 | 최대 2,400 | train에서 최대 22,600 | 최대 30,000 |

개수는 요청 상한이다. 실제 eligible 이미지 수, 다운로드 성공 수, 정확 중복 제거 결과에 따라 줄어든다.
`manifest.json`에 요청/다운로드/고유 이미지 수를 모두 남긴다. 부족분을 같은 사진 복제로 채우지 않는다.
세 소스만으로 30,000장을 확보할 수 있다고 보장하지 않으며 실제 집계 후 추가 소스를 결정한다.

- COCO는 약 253MB의 annotations ZIP을 받은 뒤 **Cat 이미지 파일만** 다운로드한다.
  캡션, 원본 이미지 URL, 사진별 라이선스 레코드, 대상 박스를 보관한다.
  공식 이미지 S3 버킷의 HTTPS 주소를 사용한다.
- Oxford 이미지는 고정 revision의 [Hugging Face 미러](https://huggingface.co/datasets/Shiym/ViT-FineTune/tree/1cc46adc2014ae0f8d67cc55a84d2022886b0364/oxford-iiit-pet)를 우선 사용한다.
  원본 공개 MD5 `5c4f3ee8e5d25df40f4fd59a7f44e54c`와 일치해야 선택된 고양이 사진을 추출한다.
  미러 실패 시 Oxford 원본 경로로 재시도한다.
  기존 머리 박스를 얼굴 크롭의 출발점으로 활용한다. CC BY-SA 4.0 메타데이터를 보관한다.
- Open Images는 공식 Cat 박스 주석을 스트리밍해 후보를 고른 뒤 해당 이미지 ID만 다운로드한다.
  train 파일 전체는 약 2.26GB이므로 이미지 후보가 목표의 2배에 도달하면 탐색을 멈춘다.
  이 후보 안에서 seed로 섞는다. **전체 train에 대한 균등 무작위 표본은 아니다.**
  별도 공식 메타데이터에서 작성자·원본 주소·라이선스를 연결한다.
- CatFLW와 슈딩이 DB는 다운로드 대상에 없다.

출처와 이용 조건은 사진별로 보존한다. Open Images는 source-listed 라이선스를 기록하며
원본 권리 상태를 자동 검증한 것으로 간주하지 않는다.

대규모 실행은 `.env`에서 다음을 바꿔 새 run으로 실행한다.

```bash
CONFIG=configs/scale.json
OUTPUT_ROOT=/data/cat-labeler/outputs/scale32b_v1
RUN_NAME=scale32b_v1
```

## 두 단계 라벨링

1. 전체 사진과 대상 박스로 기본 자세·몸 모양·착용물·장소·관계를 관찰한다.
2. 전체 사진과 원본에서 자른 얼굴/목 크롭을 새 대화로 제공해 눈·입·혀·이빨·안경·목줄을 관찰한다.
   1차 속성 답변은 2차 프롬프트에 넣지 않는다. 1차 얼굴 위치 정보는 크롭에 사용한다.
3. 같은 항목의 두 판정이 충돌하면 unknown으로 합치고 검수 플래그를 붙인다.

고양이가 여러 마리면 주석 박스 중 가장 큰 고양이를 대상으로 하며 검수 플래그를 붙인다.
얼굴 박스는 Oxford의 기존 주석 또는 1차 VLM의 위치 추정이다.
별도 얼굴 탐지기를 학습한 상태가 아니므로 VLM 크롭 좌표는 미검수로 표시한다.
크롭을 찾지 못하면 전체 사진으로 얼굴 항목을 재질문하고 `crop_geometry_check`를 남긴다.
업스케일이나 초해상도로 세부를 만들어내지 않는다.

라벨 값과 의미는 [`docs/labels.md`](docs/labels.md)에 정리했다.
전체/크롭 입력의 최대 픽셀 수는 각각 524,288 / 262,144다.
작은 이빨 등은 이 설정에서 놓칠 수 있으므로 50장 검수 후 필요하면 픽셀 상한을 바꾼 새 run과 비교한다.

## 출력과 재개

```text
$DATA_ROOT/
  downloads/                 공식 메타데이터·Oxford archive
  catalogs/                  고정된 다운로드 후보
  images/                    선택된 외부 고양이 사진
$OUTPUT_ROOT/
  preflight.json
  manifest.json              출처·상대경로·SHA256·dHash·박스·원문 캡션
  model-pin.json             첫 실행 시 main을 실제 commit SHA로 고정
  run.json                   모델·양자화·픽셀 예산·프롬프트 서명·패키지 버전
  smoke.json
  progress.json
  items/<image_id>/
    context.json
    full.json                1차 관찰
    face.json                독립 2차 관찰
    full-attempt-1.txt        실제 모델 응답
    face-attempt-1.txt
    result.json              병합된 라벨 초안·크롭 좌표·충돌·검수 플래그
    error.json               실패한 경우에만 존재
results/<RUN_NAME>/
  images-00000.jsonl         원본 출처와 이미지 체크섬
  labels-00000.jsonl         병합 라벨 초안
  observations-00000.jsonl   단계별 판정과 원문 응답
  dataset.json run.json summary.json ...
```

파일 단위 원자적 저장과 프로세스 잠금으로 중단·재실행을 처리한다.
이미지 다운로드는 실패 파일부터 재시도하고 이미 받은 정상 파일은 재사용한다.
체크섬이 알려진 Oxford archive는 `.part` 파일부터 HTTP Range로 이어받는다. 서버가 Range를 무시하면 안전하게 처음부터 쓴다.
다운로드 중 10초 간격으로 크기와 속도를 출력하고, 60초 이상 평균 64KiB/s 미만이면 다음 경로로 전환한다.
원본 체크섬 근거 — https://docs.pytorch.org/vision/main/_modules/torchvision/datasets/oxford_iiit_pet.html
모델 JSON이 스키마에 맞지 않으면 최대 두 번 요청하고 실패 원문을 남긴다.
라벨링 오류가 남으면 전체 완료로 처리하거나 자동 푸시하지 않는다.

부분 결과를 의도적으로 내보내려면 실행을 멈춘 후 다음을 사용한다.

```bash
.venv/bin/python -m cat_labeler.export --name "$RUN_NAME" --allow-partial
./publish_results.sh
```

`summary.json`의 `complete`는 처리 완료 여부이며, `training_ready`는 항상 false다.
확인 불가를 없음으로 바꾸지 않고 사람이 검수한 뒤 학습용 문장을 생성해야 한다.
현재 export에는 학습용 조합 캡션을 자동 생성하지 않는다.
정확 중복은 제거하고 dHash는 근접 중복 검수용으로 제공한다.
개체·촬영 세션 분리, 근접 중복 판정, 최종 학습/검증 분할은 별도로 확정해야 한다.

## CPU 개발 검사

```bash
uv sync --locked --extra dev
.venv/bin/python -m pytest -q
.venv/bin/python -m cat_labeler.preflight --cpu-only
```

CPU 테스트는 형식·충돌 처리·크롭·재개·export·archive 안전성을 확인한다.
CUDA 모델 추론과 A5000 실측은 서버의 `./smoke_test.sh`로 확인한다.

## 참고

- 서버 운영 형태를 참고한 레포 — https://github.com/jeehoo0507/aim-lab-test-5
- 실행 환경 분리를 참고한 레포 — https://github.com/jeehoo0507/Loop-Research-for-MaskedKD
- Qwen 모델 — https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
- 모델 입력 — https://huggingface.co/docs/transformers/model_doc/qwen3_vl
- COCO — https://cocodataset.org/
- Oxford — https://www.robots.ox.ac.uk/~vgg/data/pets/
- Open Images — https://storage.googleapis.com/openimages/web/download_v7.html
