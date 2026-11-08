# PawLy CV/ML: поиск собак по фото

Документ описывает то, что реально реализовано в коде, и отдельно — эксперимент. Он заменяет
план из [cv-research.md](./cv-research.md) (тот описывал отдельный HTTP-encoder; от него
отказались, см. «Архитектура»). Ответы ML-инженера — в [W.md](../W.md).

## 1. Утверждённые решения → где реализовано

| Решение | Реализация |
| --- | --- |
| Пользователь передаёт фото, породу, окрас | `POST /api/v2/photo-searches`: `breed_text`, `coat_color_text` обязательны; для объявлений — `listings.breed/color` |
| Порода/окрас пользователя — данные, не ground truth | хранятся verbatim (`photo_searches.breed_text`, snapshot `media_features.user_*_text`), отдельно от предсказаний; в ранжировании не участвуют |
| Неизвестные породы и метисы получают embedding | embedding не зависит от текста; «немецкая овчарка + терьер» хранится как есть |
| Результат — кандидаты, приоритет recall | top-K объявлений (v2: до 50), без жёстких фильтров по породе/окрасу |
| RGB-признаки по каждому изображению | `rgb_histogram` (64 бина) + `dominant_colors` в `media_features` |
| Без ручной разметки цветов | цветовая голова учится на алгоритмических pseudo-targets (см. §5) |
| Классы пород из открытых датасетов | таксономия = породы Oxford-IIIT Pet (dog subset) минус отложенные |
| Пользовательский crop, без сегментации | нормированный bbox (правило `crop-v1`), без bbox — всё изображение |
| 1–2 общепринятых backbone | EfficientNet-B0 `IMAGENET1K_V1` (основной), ResNet-50 `IMAGENET1K_V2` (сравнение) |
| Общий замороженный backbone + 2 головы, head-only | `app/cv/heads.py`, `app/ml/experiment.py` |
| Unknown threshold экспериментально | порог max-softmax подбирается на validation (§6) |
| Калибровка отложена, score не показывать | в API нет ни одного score/процента; внутренний `ranking_score` только в БД |
| Не-собак отклонять | dog gate SSDLite320 (COCO), `reason_code=no_dog / multiple_dogs` |

## 2. Режимы модели

| `CV_MODE` | Что работает | Чего нет |
| --- | --- | --- |
| `pretrained_features` (**по умолчанию**) | настоящий pretrained encoder, dog gate, RGB-дескриптор, индексация, поиск | breed/color предсказаний: головы не загружаются вовсе (`breed_prediction = NULL`) |
| `trained_heads` | всё выше + обученные головы из явно указанного bundle (`CV_BUNDLE_DIR`) | — |
| `mock` | только для тестов/plumbing; `provider=mock-v1`, `is_mock=true` | визуального сходства нет |

Fallback'ов нет: `trained_heads` без валидного bundle → worker в состоянии `error`
(`/health/ml` = 503, `GET /api/v1/cv/status` показывает `error_code`), поисковые задачи
повторяются и завершаются `failed: cv_unavailable`. Ни `trained_heads → pretrained_features`,
ни `→ mock` автоматически не происходит. Bundle проверяется по: SHA-256 весов голов, weights ID
и SHA-256 backbone, `embedding_space_id`, размерности, полной спецификации preprocessing,
версии RGB-дескриптора, правилу crop, хэшу таксономии, форме state_dict.

## 3. Архитектура

```
API (лёгкий образ, без torch) ──► PostgreSQL 16 + PostGIS + pgvector ◄── CV worker (torch, CPU)
   v1/v2 photo-searches                media_features (embeddings,        claim/lease/fencing
   crop, status, /health/ml            RGB, provenance, очередь)          модели в 1 потоке
```

* Модель живёт только в процессе `worker` (образ `pawly-cv`), загружается один раз и
  выполняется в отдельном потоке — не в event loop и не в Uvicorn-воркерах.
* Отдельный HTTP ML-service не добавлялся: worker уже владеет очередью, лишний сетевой hop не
  нужен.
* `media_features` — и хранилище признаков, и очередь индексации (тот же паттерн
  `FOR UPDATE SKIP LOCKED` + lease token + fenced finalize, что у `photo_searches`).
* Heartbeat worker'а пишется в `cv_worker_status`; API по нему отвечает на `/health/ml` и
  `GET /api/v1/cv/status` (агрегаты без пользовательских данных). `/health/ready` (CRUD) от ML
  не зависит.

## 4. Pipeline и preprocessing

```
canonical JPEG (normalize_image: EXIF orientation, без EXIF/GPS, sRGB-интерпретация)
  → user crop (crop-v1) → dog gate (SSDLite, только подсчёт собак, crop не меняет)
  → RGB-гистограмма по пикселям crop (до любой нормализации)
  → eval-transforms весов → frozen backbone → pooled features h → z = h / ||h||
  → [trained_heads] breed softmax + predicted RGB histogram
```

**Crop `crop-v1`.** bbox `[x_min, y_min, x_max, y_max]` ∈ [0,1], относительно канонического
изображения после EXIF-ориентации; `x_min < x_max`, `y_min < y_max`. Пиксели:
`round(v·size, 6)`, затем floor для left/top и ceil для right/bottom, clamp к изображению;
каждая сторона ≥ 32 px, иначе `invalid_crop` (HTTP 422 при создании). Хэш crop = SHA-256 от
`{rule, bbox, box, size}`.

**Backbone transforms** (детерминированные transforms выбранных весов torchvision; одна
функция `make_preprocess` для обучения и runtime; при загрузке сверяются со статической
спецификацией, расхождение = ошибка):

| Backbone | Weights | dim | resize (короткая сторона) | center crop | interpolation |
| --- | --- | --- | --- | --- | --- |
| EfficientNet-B0 | `IMAGENET1K_V1` | 1280 | 256 | 224 | bicubic |
| ResNet-50 | `IMAGENET1K_V2` | 2048 | 232 | 224 | bilinear |

Нормализация ImageNet mean/std. **Center crop обрезает края неквадратного crop** — backbone не
видит часть пользовательского crop у длинных сторон; RGB-дескриптор считается по всему crop.
Размерности проверяются в коде (`FrozenEncoder`, тесты `test_cv_models.py`/`test_cv_real.py`).

**Dog gate.** `ssdlite320_mobilenet_v3_large`, веса `COCO_V1`, категория `dog` из
`weights.meta["categories"]`. Детекции `dog` со score ≥ `CV_DOG_SCORE_THRESHOLD`
(экспериментальный default 0.5, не калиброван), дубликаты одной собаки сливаются, если одна
рамка ≥70% лежит в другой. 0 → `no_dog`, ≥2 → `multiple_dogs`. Это доменный отказ: задача
`completed`, `outcome=rejected`, без retry. Идеальной работы gate не гарантирует (§8).

**RGB-дескриптор `rgb-hist-4x4x4-v1`.** `bin = 16·(R>>6) + 4·(G>>6) + (B>>6)`, 64 бина,
доля пикселей crop, сумма = 1; плюс до 3 доминирующих бинов со средним RGB. Без маски это
характеристика **crop целиком, включая фон**, а не доказанный окрас шерсти.

**Веса.** Скачиваются при сборке образа с `download.pytorch.org`, проверяются по полному
SHA-256 (закреплён в `app/cv/weights.py`), лежат read-only в `/opt/pawly/weights`. Runtime их
не скачивает.

## 5. Формальная постановка (соответствует коду)

Вход: изображение `x` (каноническое), bbox `b` (или ∅), тексты пользователя `t_breed`,
`t_color` (сохраняются, в модели не используются). Crop `c = crop(x, b)`.

* Gate: `D(c) = {детекции dog, score ≥ τ_g}` после слияния; принять ⇔ `|D(c)| = 1`.
* Encoder (заморожен): `h = f_θ(T(c)) ∈ ℝ^d`, `z = h / ‖h‖₂` (проверка конечности и `‖h‖>0`).
* RGB: `r(c) ∈ Δ^63`, `r_j = (1/N)·Σ_p 1[bin(p) = j]`.
* Breed head: `p(c) = softmax(W_b h + β_b) ∈ Δ^{K-1}`, K — число известных пород.
* Color head: `q(c) = softmax(W_c h + β_c) ∈ Δ^63` — предсказанная гистограмма.
* Loss на батче B, `B_known = {i ∈ B : y_i ≠ unknown}`:
  `L = λ_b · (1/|B_known|) Σ_{i∈B_known} CE(W_b h_i + β_b, y_i) + λ_c · (1/|B|) Σ_{i∈B} KL(r(c_i) ‖ q(c_i))`,
  где `KL(r‖q) = Σ_j r_j (log r_j − log q_j)`, `0·log 0 = 0` (эквивалентно soft-target CE
  минус константная энтропия r). По умолчанию `λ_b = λ_c = 1`.
* Unknown masking: образцы без известной метки (неизвестная порода, метис) **не получают
  случайный класс**: они исключены из CE, но участвуют в color loss и в поиске.
* Unknown-решение: `unknown ⇔ max_k p_k(c) < τ_u`, `τ_u` — квантиль `1 − TPR` (TPR = 0.95)
  max-softmax известных validation-образцов.
* Retrieval: `s(q, i) = z_qᵀ z_i` (cosine). Оценка объявления `S(q, L) = max_{i ∈ L} s(q, i)`
  (`matched_media_id` = argmax). Сортировка по `S` убыв., ничьи — по id объявления, затем фото.
  Вычисляется точно (exact) по всем подходящим проиндексированным фото в scope.
* Опциональный fusion (версионированный, `CV_FUSION`), все члены приведены к [0, 1]:
  * `embedding-v1`: `S = cos` (baseline, по умолчанию);
  * `embedding-rgb-v1`: `S = (1−w)·(1+cos)/2 + w·HI(r_q, r_i)`, `HI(a,b) = Σ_j min(a_j, b_j)`;
  * `embedding-rgb-heads-v1` (только `trained_heads`):
    `S = (1−w−v)·(1+cos)/2 + w·HI(r_q,r_i) + v·½·(BC(p_q,p_i) + HI(q_q,q_i))`,
    `BC(a,b) = Σ_k √(a_k b_k)`. Веса `w=0.2`, `v=0.1` — конфигурация, **не** подобранные значения.
* Метрики (для запроса q с множеством релевантных `R_q ≠ ∅` и top-K `T_q(K)`):
  `Recall@K(q) = |R_q ∩ T_q(K)| / |R_q|` (доля **всех** релевантных),
  `HitRate@K(q) = 1[|R_q ∩ T_q(K)| > 0]` (найден **хотя бы один**),
  `Precision@K(q) = |R_q ∩ T_q(K)| / K`. Усреднение — по запросам с `R_q ≠ ∅`; запросы без
  релевантных считаются отдельно. В части статей HitRate называют «Recall@K» — здесь это
  разные числа.

### Почему гистограмма суммируется в 1, а multi-label scores — нет (вопрос из W.md)

1. Если предсказывать набор независимых признаков «есть чёрный», «есть белый», … , каждый
   выход — отдельная sigmoid-вероятность своего бинарного события. События не исключают друг
   друга (собака может быть и чёрной, и белой), поэтому сумма может быть 0.3 или 2.7 —
   нормировки нет и быть не должно; loss — BCE по каждому признаку.
2. Если предсказывать **распределение пикселей crop по 64 RGB-бинам**, выход — доли одного
   целого: каждый пиксель попадает ровно в один бин, поэтому `Σ_j r_j = 1`, и предсказание
   `q = softmax(·)` тоже суммируется в 1; loss — KL между распределениями.
3. В этом эксперименте выбран **второй вариант** (гистограмма + softmax + KL).
4. Варианты не смешиваются: нет BCE по цветам и нет sigmoid-выходов; «доля бина» не трактуется
   как «вероятность наличия цвета».

## 6. Эксперимент head-only

**Данные.** Oxford-IIIT Pet (CC BY-SA 4.0 по странице датасета; авторские права на фото у
владельцев). Используются только породы собак (25) и реальные метки из `annotations/list.txt`.
Bbox Oxford — это **рамки головы**, они не используются: crop policy = всё изображение
(production-crop пользователя — вся собака; это различие задокументировано и не скрывается).
Кошки Oxford используются только как размеченные не-собаки для оценки gate.

Stanford Dogs поддержан adapter'ом (`prepare-data stanford-dogs`, full-dog bbox из аннотаций,
только изображения с одной собакой), но требует явного `--accept-terms
imagenet-noncommercial-research` (изображения из ImageNet, условия — некоммерческие
исследования); хэши его архивов не закреплены, в этом этапе он **не запускался**.

**Подготовка (`prepare-data`).** Повторобезопасная загрузка/копия архивов, проверка SHA-256
(MD5 дополнительно совпали с torchvision), распаковка только обычных файлов из ожидаемых
префиксов с защитой от path traversal и ссылок. Каждое фото проходит **тот же
`normalize_image`, что и загрузка в API**. Manifest (`manifest.jsonl` + `dataset.json`):
источник, лицензия, SHA-256 файла и канонического JPEG, dHash, группа, порода, класс,
split, crop policy, версия pseudo-target. Точные дубликаты (по SHA-256 канонических байтов)
удаляются; near-duplicates (горизонтальный **и** вертикальный 64-битный dHash, Хэмминг ≤ 5 по
обоим) объединяются в группы, группа собак никогда не пересекает split'ы (проверяется при
подготовке). Первая версия с одним dHash транзитивно склеила 50 несвязанных изображений и через
такую группу 2 фото «test-unknown» пород попали в validation; это исправлено (двойной хэш +
приоритет test-unknown), и все результаты ниже получены заново на исправленном manifest. Идентификаторов отдельных животных в Oxford нет → **отсутствие identity
leakage не доказано**; изображения одной породы не считаются одной собакой.

**Split'ы.** Официальный test Oxford → test; официальный trainval → train/val по группам (15% val,
детерминированный хэш группы). 4 породы (seed 0) отложены целиком как «неизвестные»: 2 — в val
(для подбора порога), 2 — в test (для оценки). Порог и веса fusion на test не подбираются.

**Обучение (`train`).** Признаки замороженного backbone кэшируются один раз (ключ: SHA-256
manifest + `embedding_space_id` + версия дескриптора + crop rule). Это эквивалентно прогону
backbone в каждом шаге: веса заморожены, BN в eval, transforms детерминированы, аугментаций
нет. Головы: Linear (по умолчанию), AdamW lr 1e-3, wd 1e-4, batch 128, 30 эпох, seed 0;
лучшая эпоха по validation macro recall. Механика head-only проверяется тестом
`test_head_only_step_keeps_backbone_parameters_and_buffers`: `requires_grad=False`, encoder
остаётся в eval внутри `model.train()`, в optimizer только параметры голов, после шагов
обучения **параметры и буферы** (BN running stats) backbone побитово не изменились.

**Важно.** Head-only обучение не меняет embedding: поиск в `trained_heads` использует тот же
frozen embedding; вклад голов возможен только через fusion C и оценивается отдельно. Frozen
ImageNet backbone не является re-identification encoder'ом; сходство породы не доказывает,
что найдена та же собака.

## 7. Результаты (реально измерено)

Прогон 2026-11-08, всё внутри Docker, **только CPU** (Docker Desktop/WSL2, 8 vCPU, 9 GB RAM,
torch 2.14.1+cpu, без GPU), seed 0, один split, без доверительных интервалов. Артефакты:
volume `ml_artifacts` → `runs/{efficientnet_b0,resnet50}-oxford-s0/{config,history,metrics}.json`.

**Данные** (manifest `afec0d38…`): 4978 фото собак (25 пород), 2371 кошка (только для gate);
точных дубликатов 0, битых файлов 0, групп near-duplicates 42 (максимум 3 фото), групп собак,
пересекающих split'ы, 0. Таксономия — 21 порода. Train 1767; val 314 известных + 399 отложенных
(`american_pit_bull_terrier`, `keeshond`); test 2098 известных + 400 отложенных (`leonberger`,
`yorkshire_terrier`).

| | EfficientNet-B0 | ResNet-50 |
| --- | --- | --- |
| dim embedding (проверено в коде) | 1280 | 2048 |
| обучаемых параметров (головы) / backbone | 108 885 / 0 | 174 165 / 0 |
| извлечение признаков, 4978 фото, CPU | 250 с | 554 с |
| обучение голов, 30 эпох (кэш признаков) | 1.6 с | 2.0 с |
| breed: accuracy / macro recall (test, 2098) | 0.967 / 0.966 | 0.971 / 0.971 |
| худший класс (recall) | staffordshire_bull_terrier 0.831 | staffordshire_bull_terrier 0.888 |
| порог unknown τ_u (выбран на val) | 0.686 | 0.720 |
| test: принято известных / отклонено отложенных | 0.951 / 0.610 | 0.939 / 0.900 |
| color head KL / TV (pseudo-targets, test) | 0.505 / 0.360 | 0.436 / 0.333 |
| baseline «средняя train-гистограмма» KL / TV | 0.637 / 0.407 | 0.637 / 0.407 |

**Поиск по породе** (attribute retrieval; relevance = та же порода датасета; запросы = все
2498 test-фото, галерея = test без самого запроса и его near-duplicates; в среднем 115
релевантных на запрос, 0 запросов без релевантных). Веса fusion — конфигурация (0.2/0.1), на test
не подбирались.

| Метод | Backbone | Recall@10 | Recall@50 | HitRate@1 | HitRate@10 | Precision@10 | Precision@50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A embedding-v1 | EffNet-B0 | 0.088 | 0.412 | 0.972 | 0.996 | 0.948 | 0.895 |
| B embedding-rgb-v1 | EffNet-B0 | 0.086 | 0.396 | 0.962 | 0.995 | 0.931 | 0.863 |
| C embedding-rgb-heads-v1 | EffNet-B0 | 0.088 | 0.420 | 0.971 | 0.995 | 0.952 | 0.910 |
| A embedding-v1 | ResNet-50 | 0.089 | 0.436 | 0.978 | 0.994 | 0.965 | 0.942 |
| B embedding-rgb-v1 | ResNet-50 | 0.088 | 0.425 | 0.969 | 0.996 | 0.955 | 0.920 |
| C embedding-rgb-heads-v1 | ResNet-50 | 0.089 | 0.435 | 0.974 | 0.996 | 0.963 | 0.940 |

Recall@K здесь ограничен сверху величиной `K/|R_q|` (≈ 0.087 при K=10 и 115 релевантных), поэтому
малые значения Recall@10 — следствие определения, а не провал; это и показывает, почему Recall@K
и HitRate@K нельзя путать. Для запросов отложенных пород (≈200 релевантных) EffNet-B0 A:
Recall@50 = 0.244 (потолок ≈ 0.25), HitRate@1 = 0.985 — неизвестные породы находятся так же, как
известные.

**Dog gate** (200 test-собак Oxford, полные изображения, и 200 кошек): принято собак 0.895
(21 → `no_dog`), ложно принято кошек 0.050 (10/200); медиана 67 мс на CPU.

**Runtime в Compose** (CV worker, 2 потока torch, корпус 5 фото — нерепрезентативно мал):
извлечение на изображение (gate + embedding [+ головы]) 140–270 мс; выполнение поисковой задачи
130–200 мс; от `POST` до `completed` 0.7–0.9 с (включая опрос очереди раз в 1 с); покрытие
индекса 5/5. Офлайн exact cosine (numpy, галерея 2498): 1.0 мс/запрос (EffNet-B0), 3.1 мс
(ResNet-50).

**Что из этого следует.**

* Гипотеза «две головы поверх замороженного backbone минимальными средствами»: технически
  подтверждена — обе головы обучаются head-only за секунды, порода предсказывается на test Oxford
  с macro recall 0.97, цветовая голова воспроизводит pseudo-target лучше тривиального baseline,
  bundle экспортируется, валидируется и работает в `trained_heads` end-to-end (smoke пройден).
* Пользы голов **для поиска** не показано: C ≈ A (разница ≤ 0.01 при одном seed), B (RGB
  crop целиком, с фоном) немного хуже A. Поэтому по умолчанию `embedding-v1`.
* ResNet-50 немного лучше EfficientNet-B0 по всем породным метрикам и по отсеву отложенных пород,
  но в 2.2 раза медленнее при извлечении и вдвое больше по размерности. По утверждённому решению
  основной baseline остаётся EfficientNet-B0; смена backbone — одна переменная `CV_BACKBONE`
  и переиндексация.
* Числа оптимистичны для продукта: ImageNet-1k содержит большинство этих пород как классы
  (включая «отложенные» leonberger и yorkshire terrier — они неизвестны только таксономии головы,
  не backbone'у), пересечение изображений ImageNet и Oxford не исключено, метисов и фото «с улицы»
  в датасете нет.
* **Проверена породная/атрибутная близость; качество поиска той же самой собаки не измерено.**

## 8. Ограничения

* **Same-dog recall не измерен** — нет identity-разметки. Проверена породная/атрибутная
  близость; качество поиска той же самой собаки не измерено.
* Цветовая голова учится на алгоритмических pseudo-targets (гистограмма crop, включая фон).
  Её метрики — качество воспроизведения гистограммы, не точность распознавания окраса.
* Embeddings — общие ImageNet-признаки, не специализированы под собак/ракурсы/освещение.
* Dog gate проверен только на собаках и кошках Oxford; ложные срабатывания на других животных,
  игрушках, рисунках не измерялись. Порог 0.5 не калиброван.
* Exact nearest-neighbour = точное вычисление cosine по всем строкам, а не «гарантированно
  найденная собака». На большом корпусе понадобится ANN-индекс (HNSW) — сознательно не добавлен.
* Пиксели интерпретируются как sRGB: существующий upload pipeline отбрасывает ICC-профили без
  конверсии.
* Обучение на production-фото не выполняется и автоматически не включается.
* Текстовый поиск/boost по породе и окрасу не реализован: нормализованные поля (`*_norm`)
  сохраняются для него, но в ранжировании текст не участвует (`search_meta.user_text_used_for_ranking=false`).
* Stanford Dogs adapter реализован, но не запускался (условия ImageNet + незакреплённые хэши).
* GPU-override (`compose.gpu.yaml`) не проверялся — на машине нет GPU.

## 9. Хранение и lineage

| Данные | Где |
| --- | --- |
| Канонические изображения | существующее media storage (volume `media_data`) |
| Текст пользователя (raw + нормализованный для поиска) | `photo_searches.breed_text/coat_color_text(_norm)`; объявления — `listings.breed/color` + snapshot в `media_features.user_*_text` |
| Embedding (pgvector), RGB, предсказания голов, provenance | `media_features`: `pipeline_id`, `embedding_space_id`, backbone, weights ID, `preprocessing_id`, `descriptor_version`, `gate_id`, `bundle_id`, `dimension`, crop (`crop_bbox`, `crop_box`, `crop_hash`, `crop_rule`), `input_sha256`, статус/причина/ошибка, timestamps |
| Результат поиска | `photo_searches` (+ `outcome`, `reason_code`, `cv_mode`, `pipeline_id`, `embedding_space_id`, `fusion_version`, `search_meta`), `photo_search_results` (`ranking_score` — только внутренний) |
| Эксперименты | volume `ml_artifacts`: `datasets/`, `features/`, `runs/<run>/{config,history,metrics}.json`, `bundles/<id>/` |

Инварианты в БД: `vector_dims(embedding) = dimension` для `indexed`; уникальность
`(media_id, crop_hash, pipeline_id)`; не более одной текущей строки на `(media, pipeline)`.
Поиск сравнивает только строки активного `pipeline_id` с тем же `embedding_space_id` и
размерностью (1280 и 2048 никогда не смешиваются). Смена crop → старые строки получают
`superseded_at` (история сохраняется), новая версия считается заново; возврат к прежнему crop
«оживляет» старую строку без пересчёта. Удалённые фото исключаются из выдачи сразу (фильтр
`media_assets.status='ready'`), а `cleanup-media` физически удаляет их признаки.

## 10. Индексация

* Worker раз в `CV_INDEX_SWEEP_SECONDS` (10 с) ставит в очередь готовые фото объявлений без
  текущей строки активного pipeline; затем обрабатывает их (поисковые задачи — в приоритете).
* Короткие транзакции; inference — вне транзакции; lease/fencing; до 3 попыток при временных
  ошибках; доменный отказ (`no_dog` и т.п.) — финальный статус `rejected`, без retry.
* Опубликованное объявление попадает в выдачу, как только его фото `indexed`; скрытое,
  архивное, удалённое — исключается немедленно (проверка и при поиске, и при чтении).
* Запрос не сравнивается сам с собой: query-фото имеют `purpose='query'` и исключаются явно.
* Пустой корпус → `completed`, `outcome=candidates`, `results=[]`, `search_meta.eligible_listings=0`.

```sh
docker compose exec worker python -m app.cli model-status
docker compose exec worker python -m app.cli index-existing-media   # фото, загруженные ранее
docker compose exec worker python -m app.cli reindex [--failed-only] # пересчёт активного pipeline
```

## 11. Команды (всё внутри Docker)

```sh
docker compose up -d --wait                       # API + CV worker (pretrained_features)
docker compose --profile ml build ml
docker compose --profile ml run --rm ml python -m app.ml prepare-data oxford-pets
docker compose --profile ml run --rm ml python -m app.ml train --backbone efficientnet_b0 --run-name effb0-s0
docker compose --profile ml run --rm ml python -m app.ml train --backbone resnet50 --run-name r50-s0
docker compose --profile ml run --rm ml python -m app.ml evaluate --run effb0-s0
docker compose --profile ml run --rm ml python -m app.ml export --run effb0-s0 --name effb0-s0-v1
docker compose --profile ml run --rm ml python -m app.ml check-bundle /data/ml/bundles/effb0-s0-v1
```

Первый `prepare-data` скачивает ~810 MB с `thor.robots.ox.ac.uk`. Уже скачанные архивы можно
подать read-only: `-v /path/to/archives:/archives:ro ... prepare-data oxford-pets
--archives-dir /archives` (хэши проверяются всё равно). GPU (опционально, не проверено на этой
машине — GPU не было): `docker compose -f compose.yaml -f compose.gpu.yaml --profile ml run
--rm ml python -m app.ml train --device cuda ...` (CUDA-колёса cu130 ставятся при сборке и не
закреплены lock-файлом).

**Активация bundle / rollback.**

```sh
CV_MODE=trained_heads CV_BUNDLE_DIR=/data/ml/bundles/effb0-s0-v1 docker compose up -d worker api
docker compose exec worker python -m app.cli model-status   # bundle.valid = true, index counts
# rollback: вернуть прежний режим; строки старого pipeline не удалялись и сразу доступны
docker compose up -d worker api
```

Новый bundle = новый `pipeline_id`: его индекс строится в фоне (sweep), пока покрытие неполное,
выдача содержит только уже пересчитанные фото (видно в `GET /api/v1/cv/status`). Строки разных
pipeline сосуществуют, поэтому откат мгновенный. Bundle экспортируется атомарно
(`<name>.partial` → rename) и никогда не перезаписывается.

## 12. Существующий том БД

Образ `pawly-db` = `postgis/postgis:16-3.4` + пакет `postgresql-16-pgvector 0.8.0` (из архива
PGDG; серверные пакеты PostgreSQL/PostGIS не меняются, существующий `pg_data` совместим).
Для новых volume расширение создаёт init-скрипт под bootstrap-суперпользователем. Для
существующего volume:

```sh
docker compose build db && docker compose up -d --wait db
docker compose run --rm migrate      # создаёт vector, если роль миграции — суперпользователь
```

В Compose `POSTGRES_USER` — суперпользователь, поэтому это работает без ручных шагов (проверено
на существующем volume этого репозитория). Если роль миграции не суперпользователь (pgvector
0.8.0 не «trusted»), миграция остановится с понятным сообщением; DBA выполняет один раз
`CREATE EXTENSION IF NOT EXISTS vector;`.
