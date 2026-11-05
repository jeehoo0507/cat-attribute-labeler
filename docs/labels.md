# 라벨 규약 v1

대상 고양이별로 관찰한다. 출력은 영어 enum과 짧은 한국어 근거로 저장한다.
고정값 이외의 표현은 JSON validation에서 거절한다.

| 항목 | 허용값 |
| --- | --- |
| 기본 자세 | sitting, standing, prone, side_lying, supine, other, unknown |
| 앞발 접기·몸 말기·앞다리 뻗기 | 각각 present, absent, unknown |
| 왼쪽/오른쪽 눈 | 각각 open, partly_open, closed, unknown |
| 입 | closed, slightly_open, wide_open, unknown |
| 혀 | outside_lips, inside_mouth_only, not_visible, unknown |
| 이빨·발바닥·리드줄 | 각각 visible, not_visible, unknown |
| 안경·칼라·스카프·머리 착용물·옷 | 각각 present, absent, unknown |
| 옷 종류·색·무늬 | 관찰한 자유 텍스트. 원문을 보존하고 검수 후 정규화 |
| 배경 | 관찰한 물체/장소 이름 목록 |
| 관계 | object + on/in/under/beside/in_front_of/held_by + 근거 |
| 상호작용 | 직접 확인한 접촉·동작의 짧은 설명 |

`unknown`은 가림·잘림·흐림·해상도 부족·애매함이다.
`absent`는 관련 부위를 충분히 확인하고 없다고 판정한 경우다.
`not_visible`은 해당 부위를 확인했으나 혀/치아 등이 드러나지 않는 경우다.
예를 들어 얼굴이 통째로 가려졌다면 tongue/teeth는 unknown이다.

눈 좌우는 고양이 신체 기준이다. 반대쪽 눈이 안 보이면 unknown으로 둔다.
자는 중/행복함/졸림/하품/걷는 중 등의 의미를 눈·입·한 프레임만으로 추측하지 않는다.
관계는 대상 고양이와 연결한다. 주변에 계단이 있다는 이유로 on stairs를 붙이지 않는다.

full/face 독립 관찰이 직접 충돌하면 최종값을 unknown으로 두며 양쪽 원문을 보관한다.
코드가 강제하는 검사는 JSON 구조·허용값·좌표 범위와 판정 충돌이다.
시각적 정답 여부는 사람이 원본/크롭을 확인해야 한다.

검수자는 result.json의 labels를 원본과 대조해 수정하고, human_verified를 표시하기 전에
본인 식별자·검수 시각·수정 근거를 별도 기록한다. 초기 50장 전체와 작은 특징을 우선 검수한다.
서버 자동 처리 단계에서는 human_verified를 부여하지 않는다.
