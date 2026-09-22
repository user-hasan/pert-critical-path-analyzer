# PERT & Critical Path Analyzer — وثيقة البنية (Architecture) بالعربية

**Version (الإصدار):** 1.0 — **Phase (المرحلة):** 8+ — **Supervisor (المشرف):** Dr. Adel Al-Afeery

> هذه ترجمة/صياغة عربية لوثيقة `ARCHITECTURE.md` مع تحديث يعكس الحالة المنفذة فعلياً
> (Implemented State) للمشروع. المصطلحات الأساسية والعناوين المهمة مكتوبة
> **بالإنجليزية** وبجانبها المعنى **بالعربية بين قوسين**.

---

## 1. High-Level Architecture (البنية عالية المستوى)

```
┌─────────────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER (طبقة العرض)                     │
│  Analysis │ Understanding │ Review │ Validation │ Results │ Build│
│              PySide6 + Qt Signals/Slots + QThread Workers        │
├─────────────────────────────────────────────────────────────────┤
│           APPLICATION LAYER (طبقة التطبيق)                       │
│  EndToEndAnalyzer (المنفذ الفعلي ذو 12 مرحلة)                    │
│  AnalysisOrchestrator (غلاف إداري قديم) + ReviewWorkflow         │
├─────────────────────────────────────────────────────────────────┤
│              DOMAIN LAYER (طبقة المجال)                          │
│  CPM Engine │ PERT Engine │ Graph Builder │ Validation Engine    │
│  (خوارزميات خالصة بلا اعتماديات خارجية)                          │
├─────────────────────────────────────────────────────────────────┤
│         COMPUTER VISION LAYER (طبقة الرؤية الحاسوبية)            │
│  Preprocessing │ Shape Detection │ Arrow Detection │ OCR Engine  │
│  Classification │ Spatial Association │ Reconstruction           │
├─────────────────────────────────────────────────────────────────┤
│          INFRASTRUCTURE LAYER (طبقة البنية التحتية)              │
│  Config Manager │ Reporting/Export │ Persistence │ Benchmark     │
└─────────────────────────────────────────────────────────────────┘
```

**Dependency Rules (قواعد الاعتماد):**

- الطبقات العليا تعتمد على السفلى فقط (Upper layers depend on lower layers only).
- الطبقات السفلى لا تعتمد أبداً على العليا (Lower layers never depend on upper layers).
- طبقة المجال `Domain` خالية تماماً من اعتماديات الأطر الخارجية (Zero framework dependencies).
- طبقة `CV` تطبق الواجهات (Interfaces) التي تعرفها طبقة `Domain`.

---

## 2. Layered Architecture (البنية الطبقية)

| Layer (الطبقة) | Responsibility (المسؤولية) | Depends On (تعتمد على) |
|---|---|---|
| **Presentation (العرض)** | الواجهة الرسومية والتفاعل مع المستخدم (GUI) | Application Layer |
| **Application (التطبيق)** | تنسيق خط الأنابيب وإدارة الحالة (Orchestration) | Domain Layer |
| **Domain (المجال)** | منطق العمل والخوارزميات ونظرية الرسوم (Business Logic) | Core Models فقط |
| **Infrastructure (البنية التحتية)** | الرؤية الحاسوبية و OCR والحفظ والإعدادات (CV/OCR/Persistence) | Core Models |

---

## 3. Module Responsibilities (مسؤوليات الوحدات)

### 3.1 Core Models (النماذج الأساسية) — `pert_analyzer/core/models.py`

- كل هياكل البيانات معرفة كـ `Typed Dataclasses` (أصناف بيانات مكتوبة الأنواع).
- حاويات بيانات خالصة بلا أي منطق عمل (Pure data containers, no business logic).
- مستخدمة من **كل** الطبقات (Used by ALL layers).

### 3.2 Core Interfaces (الواجهات الأساسية) — `pert_analyzer/core/interfaces.py`

- أصناف مجردة `ABC` تعرف العقود (Contracts): `OCREngine` ،`ShapeDetector` ،`ArrowDetector` ،`DiagramClassifier` ،`GraphBuilder` ،`AnalysisEngine` ،`NetworkVisualizer`.

### 3.3 CV Pipeline (خط الرؤية الحاسوبية) — `pert_analyzer/cv/`

- **Preprocessor (المعالج المسبق):** تحويل رمادي، عتبة، إزالة ضجيج، تصحيح ميلان.
- **ShapeDetector (كاشف الأشكال):** دوائر ومستطيلات ومضلعات عبر `findContours`.
- **ArrowDetector (كاشف الأسهم):** خطوط ورؤوس أسهم عبر `HoughLinesP`.
- **DiagramClassifier (مصنف المخططات):** تمييز `AON` مقابل `AOA`.
- **SpatialAssociator (الرابط المكاني):** ربط نصوص OCR بالعناصر المرئية.

### 3.4 OCR Engine (محرك التعرف الضوئي) — `pert_analyzer/cv/ocr_engine.py`

- واجهة مجردة تدعم `Tesseract` و `EasyOCR` و `PaddleOCR` (فعلياً: `TesseractOCREngine` + `MockOCREngine` للاختبارات).
- استخراج النص مع صناديق الإحاطة (Bounding Boxes) ودرجات الثقة (Confidence).
- دعم الإنجليزية والعربية (English + Arabic).

### 3.5 Analysis Engines (محركات التحليل) — `pert_analyzer/analysis/`

- **CPMEngine:** حسابات المسار الحرج (Critical Path Method).
- **PertEngine:** حسابات بيرت بتقديرات `O/M/P` واحتمالية الإنجاز.

### 3.6 Graph Module (وحدة الرسم) — `pert_analyzer/graph/`

- **GraphBuilder:** بناء رسم `NetworkX` من العناصر المكتشفة (بنمط `Fluent API`).
- **NetworkXAdapter:** عزل مكتبة `NetworkX` عن بقية الكود.

### 3.7 Validation (التحقق) — `pert_analyzer/validation/` + `pipeline/human_review.py`

- تحقق هيكلي (Structural) وتحقق بيانات (Data) وتقييم ثقة (Confidence) مع تقارير مشاكل (Issues).

### 3.8 GUI (الواجهة الرسومية) — `pert_analyzer/gui/` (مكتبة `PySide6`)

- ست صفحات: `Analysis` (التحليل) و `Understanding` (الفهم) و `Review` (المراجعة) و `Validation` (التحقق) و `Results` (النتائج) و `NetworkBuilder` (باني الشبكة اليدوي).
- الاتصال مع طبقة التطبيق عبر إشارات Qt (Signals/Slots) وخيوط عمل (QThread Workers) حتى لا تتجمد الواجهة.

### 3.9 Persistence & Reporting (الحفظ والتقارير)

- `persistence/`: حفظ/تحميل الرسم بصيغ `JSON` و `CSV`.
- `reporting/`: بناء التقرير (Qt-free) والتصدير إلى `PDF` و `Excel` و `JSON` و `CSV` عبر `ExportManager`.

---

## 4. Data Models (نماذج البيانات)

### 4.1 Geometry Models (النماذج الهندسية)

- `Point (نقطة)`: إحداثيات `x, y` مع `distance_to`.
- `BoundingBox (صندوق الإحاطة)`: `x, y, width, height` مع `center` و `contains_point` و `overlaps`.
- `Arrow (سهم)`: `start` و `end` و `has_arrowhead` و `confidence` و `length/midpoint`.

### 4.2 Visual Element Models (نماذج العناصر المرئية)

- `DetectedShape (شكل مكتشف)`: النوع (`circle/rectangle/polygon`) والكونتور والثقة.
- `OCRResult (نتيجة OCR)`: النص والثقة والقيمة الرقمية المحللة `parsed_value` (تُستخدم للمدد).

### 4.3 Semantic Models (النماذج الدلالية)

- `Node (عقدة)`: حدث أو نشاط حسب النمط (`event/activity`).
- `Activity (نشاط)`: `activity_id` و `duration` وتقديرات بيرت (`optimistic/most_likely/pessimistic`) والعقدتان الطرفيتان، مع `expected_time = (O + 4M + P) / 6` و `variance = ((P − O) / 6)²`.
- `Dependency (اعتمادية)`: `source` → `target` ونوع العلاقة (الافتراضي `finish_to_start`).

### 4.4 GraphModel (نموذج الرسم) — القلب الدلالي للمشروع

- `diagram_type` بقيم `AON` (الأنشطة على العقد) أو `AOA` (الأنشطة على الأسهم).
- قواميس `nodes` و `activities` وقائمة `dependencies` مع `source/sink` وثقة إجمالية.
- يمنع التكرار (Duplicate) برمي `ValueError` عند الإضافة المكررة.

### 4.5 Analysis Models (نماذج التحليل)

- `ActivityAnalysis`: لكل نشاط `ES/EF/LS/LF` (البدء/الانتهاء المبكر والمتأخر) و `total_float` (الفائض الكلي) و `free_float` (الفائض الحر) و `is_critical`.
- `AnalysisResult`: مدة المشروع `project_duration` والمسار الحرج `critical_path` وكل المسارات الحرجة `critical_paths` والتباين والانحراف المعياري.

### 4.6 Validation Models (نماذج التحقق)

- `ValidationIssue`: كل مشكلة لها `severity` بخطورة (`error/warning/info`) و `category` و `message` واقتراح `suggestion`.
- `ValidationResult`: `is_valid` وعدادات الأخطاء ودرجة الثقة.

### 4.7 Project (المشروع)

- حاوية المراحل كلها: `diagram` (مخرجات CV الخام) و `graph` (النموذج الدلالي) و `analysis` و `validation` مع `status` بقيم (`draft/analyzing/review/complete/error`).

---

## 5. End-to-End Pipeline (خط الأنابيب الشامل) — `pipeline/analyzer.py`

المنفذ الفعلي هو `EndToEndAnalyzer` (أما `AnalysisOrchestrator` في `application.py` فهو غلاف إداري قديم بدوال Placeholder):

```
Image File (ملف الصورة)
  │ 01. Load Image (تحميل الصورة عبر cv2.imread)
  │ 02. Preprocess (المعالجة المسبقة)
  │ 03. Shape Detection (كشف الأشكال)
  │ 04. Classification (تصنيف AON/AOA/UNKNOWN)
  │ 05. Arrow Detection (كشف الأسهم)
  │ 06. OCR (تعرف مزدوج: كامل الصورة + مناطق العقد RegionOCR)
  │ 07. Text Association (الربط المكاني نص↔شكل/سهم)
  │ 08. Semantic Reconstruction (إعادة البناء الدلالية)
  │ 09. Graph Build (بناء الرسم مع بوابة _collect_graph_blockers)
  │ 10. Validate + CPM (التحقق ثم CPM — النوع AOA غير مدعوم NOT_SUPPORTED)
  │ 11. Human Review Analysis (تحليل الحاجة لمراجعة بشرية)
  │ 12. Final Status + Debug Export (الحالة النهائية والتصدير التشخيصي)
  ▼
PipelineResult (النتيجة)
```

- كل مرحلة تسجل `StageResult` (الحالة/الخطأ/التحذير) و `StageTiming` (التوقيت) مع دعم `Progress Callback` (رد النداء للتقدم).
- `PipelineResult.status` يأخذ إحدى القيم: `SUCCESS` (نجاح) أو `REVIEW_REQUIRED` (يلزم مراجعة) أو `FAILED` (فشل) أو `NOT_SUPPORTED` (غير مدعوم).
- تُرفع راية `review_required=True` عندما: ثقة أي نشاط `< 0.5`، أو نشاط بلا دليل OCR، أو مدة مفقودة، أو غموض في إعادة البناء، أو مخطط `AOA`.
- `ReviewCorrectionAPI`: تصحيح برمجي بلا إعادة تشغيل CV (`correct_activity_duration/id` ،`correct_dependency_source/target` ،`remove_false_detection` ،`add_missing_dependency` ،`mark_dummy_activity` ثم `reanalyze`).
- `DebugExporter`: يصدّر صوراً مرقمة (`01_original … 07_ocr`) وملف `result.json` لتشخيص فشل CV.
- سطر الأوامر (CLI): `python -m pert_analyzer.pipeline analyze-image "diagram.png" [--output result.json] [--debug] [--json]`

---

## 6. Image Preprocessing (المعالجة المسبقة للصور) — `cv/preprocessing.py`

`ImagePreprocessor` ينتج **8 تمثيلات (Representations)** لأن كل مهمة لاحقة تحتاج مدخلها الأمثل:

| Representation (التمثيل) | Use Case (الاستخدام) |
|---|---|
| `original` | المرجع و OCR |
| `grayscale` | كشف الأشكال |
| `denoised` | ما قبل الكشف (مخفف الضجيج) |
| `contrast_enhanced` | استخراج الميزات (CLAHE) |
| `binary / adaptive_binary` | كشف الأشكال (عتبة Otsu/تكيفية) |
| `edges` | كشف الخطوط/الأسهم (Canny/Sobel) |

- `CoordinateMapping (تخطيط الإحداثيات)`: يحفظ `scale/offset/rotation` لتحويل أي كشف إلى إحداثيات الصورة الأصلية (`to_original` / `to_processed` / `scale_bbox`) — كل النتائج محفوظة بإحداثيات الأصل.
- كل الخطوات قابلة للضبط عبر `PreprocessingConfig` (حجم البلور، طريقة العتبة، CLAHE، Canny، Deskew).
- مبادئ: الحفاظ على نسبة الأبعاد (Aspect Ratio)، عدم تشويه المخطط، الأصل محفوظ دائماً، لا منطق كشف هنا.

---

## 7. Shape Detection & Classification (كشف الأشكال والتصنيف)

```
Binary Image → findContours → فلترة الحدود → تصنيف الشكل
(مستطيل/دائرة/مضلولة عبر approxPolyDP + Circularity)
→ Confidence Scoring (درجة الثقة) → إزالة التكرار بـ IoU
→ CandidateNode (عقدة مرشحة: مستطيل=activity، دائرة=event)
```

**DiagramClassifier (مصنف المخططات)** — تصنيف احتمالي بالأدلة (Evidence-based):

| Evidence Factor (عامل الدليل) | AON Score | AOA Score |
|---|---|---|
| نسبة المستطيلات | `+rect_ratio × 0.4` | — |
| نسبة الدوائر | — | `+circle_ratio × 0.4` |
| 3+ مستطيلات | `+0.3` | — |
| 3+ دوائر | — | `+0.3` |
| تجانس الأحجام | `+0.15` | `+0.15` |

- `AON`: المستطيلات غالبة وثقة `> 0.3` — `AOA`: الدوائر غالبة — وإلا `UNKNOWN`.

---

## 8. Arrow Detection (كشف الأسهم) — `cv/arrow_detection.py`

الكاشف بصري فقط وينتج **دليلاً (Evidence) لا قراراً دلالياً** — أي أن `start/end` مرتبة هندسياً فقط والاتجاه الحقيقي في `arrowhead_point`:

```
HoughLinesP → فلترة خطوط الحدود → فلترة خطوط محيط الأشكال
→ كشف رأس السهم (مثلثية Triangularity + حدة القمة Apex)
→ استنتاج الاتجاه (Direction: RIGHT/LEFT/UP/DOWN/DIAGONAL)
→ دمج القطع الخطية (Collinear Merging) → ربط العقد (Node Association)
→ Confidence Scoring (درجة الثقة)
```

تركيبة الثقة: رأس سهم `+0.3`، خط طويل `+0.2`، ارتباط بالعقدتين `+0.2`، استقامة عالية `+0.2`.

> ملاحظة توافقية: مخرجات `HoughLinesP` شكلها `(N,1,4)` في بناءات OpenCV القديمة و `(N,4)` في الحديثة — الكود يعالج الشكلين عبر `reshape(-1)`.

---

## 9. OCR & Text Association (التعرف الضوئي والربط المكاني)

### OCR Engine Abstraction (تجريد محرك التعرف)

- `OCREngineBase (ABC)` يطبقها `MockOCREngine` (للاختبارات بلا اعتماديات) و `TesseractOCREngine` (محول `pytesseract`، اختياري) عبر مصنع `create_ocr_engine(name)`.

### Text Processing (معالجة النصوص)

- **TextNormalizer (المنظّم):** توحيد `NFKC`، تحويل الأرقام العربية-الهندية (`٠١٢٣٤٥٦٧٨٩` و `۰۱۲۳۴۵۶۷۸۹` ← `0-9`)، دمج المسافات، حذف محارف العرض الصفرية — مع حفظ النص الخام.
- **NumericExtractor (مستخرج الأرقام):** صحيحة وعشرية وفواصل آلاف وأزواج شبه PERT (`3/6`).
- **TextClassifier (المصنف):** أنواع `ACTIVITY_ID` (مثل `A1`) و `LABEL` (كلمات) و `NUMERIC` (أرقام) و `UNKNOWN` — تصنيف بالأدلة لا بالإجبار.

### Spatial Association (الربط المكاني) — `SpatialAssociator`

- نمط `AON`: النص **داخل** الشكل (احتواء Containment).
- نمط `AOA`: النص **قرب** السهم (مسافة عمودية + إسقاط + قرب من المنتصف).
- الأوزان: احتواء `0.4` + مسافة `0.3` + تداخل `0.2` + تمركز `0.1`، والغموض (Ambiguity) عندما يقل فرق أفضل نتيجتين عن `0.15`.
- الثقات الأربع منفصلة ولا تُدمج في رقم واحد غامض: ثقة OCR وثقة الشكل وثقة السهم وثقة الربط.

---

## 10. Semantic Reconstruction (إعادة البناء الدلالية) — `cv/reconstruction.py`

`ReconstructionEngine` هو **الجسر (Bridge)** بين الكشف منخفض المستوى ونموذج الرسم — يستهلك نتائج المراحل السابقة ولا ينفذ أي عملية CV:

- استراتيجية `AON`: مستطيل ← `ReconstructedActivity` (نشاط)، دائرة ← `ReconstructedEvent` (حدث)، سهم ← `ReconstructedDependency` (اعتمادية)، والنص داخل الأشكال يعطي المعرف والمدة.
- استراتيجية `AOA`: دائرة ← حدث، سهم ← نشاط (يُشتق التسلسل عبر الأحداث المشتركة)، والنص قرب الأسهم.
- كل عنصر يحمل `EvidenceTrace (أثر الدليل)`: المرحلة المصدر والمعرفات المصدرية والمساهمة في الثقة — أي provenance (تتبع المصدر) كامل.
- دمج الثقات بمتوسط موزون: القيمة العظمى بوزن `0.4` والبقية `0.6`.
- `AmbiguityIssue (مشاكل الغموض)`: أنواع `TEXT_NO_MATCH` و `ARROW_NO_SOURCE/TARGET` و `SHAPE_NO_TEXT` و `DUPLICATE_LABEL` و `MISSING_DURATION` و `ISOLATED_NODE`.
- `ReconciliationEngine (محرك المصالحة)`: تحليل اتصالية، تحديد `START/FINISH`، فلترة الإيجابيات الكاذبة (False Positives)، توحيد المعرفات والمدد، ثم تشغيل CPM إن أمكن.
- التحويل النهائي: `ReconstructedDiagram.to_graph_model()` ← رسم `GraphModel` جاهز لـ `CPMEngine`.

---

## 11. Diagram Types (نوعا المخططات)

### AON Detection (معايير مخطط الأنشطة على العقد)

- المستطيلات غالبة، النص داخل الأشكال (أسماء/مدد)، الأسهم تربط الأشكال مباشرة، لا عقد دائرية.

### AOA Detection (معايير مخطط الأنشطة على الأسهم)

- الدوائر (عقد الأحداث المرقمة) غالبة، النص قرب الأسهم (معرف النشاط ومدته)، أسهم متقطعة للأنشطة الوهمية (Dummy Activities) ذات المدة الصفرية.

### AON Representation (تمثيل AON)

- العقد = أنشطة (تحمل ID والمدة)، الحواف = اعتماديات.

### AOA Representation (تمثيل AOA)

- العقد = أحداث مرقمة، الحواف = أنشطة (تحمل ID والمدة).

---

## 12. Graph Reconstruction (إعادة بناء الرسم) — 5 مراحل

1. **Element Classification (تصنيف العناصر):** تجميع الأشكال حسب النوع ومطابقة عناقيد OCR مكانياً.
2. **Node Construction (بناء العقد):** إنشاء العقد من الأشكال مع التسميات من OCR.
3. **Edge Construction (بناء الحواف):** تتبع مسارات الأسهم وربط تسميات الأنشطة وكشف الوهمية.
4. **Graph Assembly (تجميع الرسم):** بناء `DiGraph` في NetworkX وتحديد عقدتي المصدر والمصب.
5. **Graph Validation (التحقق):** الرسم بلا دورات `DAG` ومتصل والبيانات مكتملة.

---

## 13. Validation Strategy (استراتيجية التحقق)

- **Structural (هيكلي):** الرسم `DAG`، مصدر واحد ومصب واحد، كل العقد قابلة للوصول من المصدر وواصلة للمصب، لا مكونات منفصلة.
- **Data (بيانات):** كل نشاط له مدة موجبة، معرفات فريدة، لا وصلات مكررة، وقيم بيرت مرتبة `O ≤ M ≤ P`.
- **Confidence (ثقة):** ثقة الكشف و OCR فوق العتبة، لا ارتباطات غامضة، كل عنصر له بيانات.
- كل مشكلة تحمل `severity` و `category` واقتراحاً ومعرف العنصر لإبرازه في الواجهة.

---

## 14. Confidence Scoring (تقييم الثقة)

```
overall_confidence = shape×0.2 + arrow×0.2 + ocr×0.3 + association×0.15 + classification×0.15
```

- **High (عالية) ≥ 0.85:** قبول تلقائي مع مراجعة اختيارية.
- **Medium (متوسطة) 0.60–0.84:** تتطلب مراجعة المستخدم.
- **Low (منخفضة) < 0.60:** تتطلب تصحيحاً يدوياً.

---

## 15. CPM / PERT Engines (محركا المسار الحرج وبيرت)

### CPMEngine — `analysis/cpm_engine.py` (يعمل على `DAG` عقدُه الأنشطة)

1. تحقق: رسم فارغ `EmptyGraphError`، مدة سالبة `InvalidDurationError`، صفر لغير الوهمي `MissingDurationError`، دورة `CyclicGraphError`.
2. **Forward Pass (المرور الأمامي)** بترتيب طوبولوجي: `ES = max(EF السابقات)` و `EF = ES + duration`.
3. **Backward Pass (المرور الخلفي)** بعكسه: مدة المشروع `= max(EF)` و `LF = min(LS اللاحقات)` و `LS = LF − duration`.
4. **Floats (الفوائض):** الكلي `LS − ES` والحر `min(ES اللاحقات) − EF`، والحرج حيث الفائض ≈ `0` (بعتبة `1e-9`).
5. كل المسارات الحرجة عبر subgraph الحرجة فقط + `all_simple_paths` مع فلترة صارمة `sum(duration) == project_duration`.

### PertEngine — `analysis/pert_engine.py` (طبقة مستقلة لا تعدل الرسم)

- `TE = (O + 4M + P) / 6` و `Variance = ((P − O) / 6)²` لكل نشاط (الناقص ← `REVIEW_REQUIRED` ولا يُخترع صفر).
- نفس المرورين الأمامي/الخلفي بمدد `TE`، وتباين المشروع = مجموع تباينات المسار الحرج الأول.
- **Completion Probability (احتمالية الإنجاز):** `Z = (T − expected) / sd` ثم `CDF = 0.5 × (1 + erf(Z/√2))` بمكتبة `math` فقط بلا محاكاة.
- نقطة التكامل: المحركان يطبقان واجهة `AnalysisEngine` ويأخذان `GraphModel` ويعيدان `AnalysisResult`.

---

## 16. GUI Architecture (بنية الواجهة) — `PySide6`

```
Header (العنوان + مؤشر سير العمل + شارة الحالة)
Body: Sidebar (شريط جانبي 200px) | QStackedWidget (6 صفحات)
StatusBar (التقدم والرسائل)
```

| Page (الصفحة) | Role (الدور) |
|---|---|
| `AnalysisPage` | رفع الصورة ومعاينتها (Zoom ×8/Sحب/لصق) ثم التحليل مع لوحة تقدم المراحل |
| `UnderstandingPage` | كانفس إعادة البناء الأولية (عقد مؤكدة/معلقة) قبل المراجعة |
| `ReviewPage` | مركز المراجعة البشرية (فئات/قائمة/تفاصيل + Apply & Validate) |
| `ValidationPage` | قائمة تحقق من 8 بنود + جاهزية CPM + فتح العنصر في المراجعة |
| `ResultsPage` | لوحة النتائج بخمسة تبويبات (Overview/Network/Activities/Critical Paths/PERT) + تصدير |
| `NetworkBuilderPage` | بناء يدوي مستقل عن الصورة (جدول + مشهد حي + حساب CPM) |

- **Communication Pattern (نمط الاتصال):** الواجهة ← التطبيق: استدعاءات — التطبيق ← الواجهة: إشارات Qt — العمال `AnalysisWorker/ApplyReviewsWorker/CpmWorker/ExportWorker` خيوط `QThread` لا تلمس الواجهة.
- **GuiSession (الجلسة):** مالك الحالة الوحيدة (`AppState` بتسع حالات) تشتق الجاهزية (`validation_status/cpm_eligible`) وتبني التقرير نفسه المستخدَم في التصدير.
- الثيمات `gui/themes/`: لوحة داكنة (`ACCENT #4F8CFF`) وخطوط `Segoe UI` ومسافات موحدة.

---

## 17. Testing & CI (الاختبار والتكامل المستمر)

- **Unit Tests (اختبارات الوحدة):** النماذج، حسابات CPM/PERT، بناء الرسم، قواعد التحقق، الإعدادات.
- **Integration Tests (اختبارات التكامل):** وصلات المراحل، تدفق `CV → OCR → Graph`، تكامل محركات التحليل.
- **CV Component Tests:** دقة كشف الأشكال/الأسهم/OCR/التصنيف على صور مرجعية.
- **End-to-End Tests:** خط كامل صورة ← تحليل لمخططات `AON` و `AOA` مع معالجة الأخطاء.
- **Test Data (بيانات الاختبار):** مخططات مرجعية (`tests/test_data/reference_diagrams`) وتعليقات الحقيقة الأرضية (Ground Truth).
- **CI:** ملف `.github/workflows/ci.yml` يشغل `pytest` على `Python 3.11/3.12` — الحالة الحالية **1431 passed, 1 skipped**.

---

## 18. Future Extensibility (قابلية التوسع مستقبلاً)

- **Extension Points (نقاط التوسعة):** محرك OCR جديد ← طبّق `OCREngine`؛ نوع مخطط جديد ← وسّع `DiagramClassifier` و `GraphBuilder`؛ محرك تحليل جديد ← طبّق `AnalysisEngine`؛ مرئيات جديدة ← طبّق `NetworkVisualizer`؛ صيغة تصدير أو قاعدة تحقق جديدة.
- **Scalability (قابلية التوسع):** مخططات ضخمة (+100 نشاط)، معالجة دفعات (Batch)، معالجة CV متوازية.

---

## 19. Risk Assessment (تقييم المخاطر)

| Risk (الخطر) | Mitigation (التخفيف) |
|---|---|
| دقة OCR للنصوص المختلطة عربي/إنجليزي | محركات OCR متعددة + مراجعة بشرية |
| تنوع أساليب المخططات في كشف الأشكال | عتبات تكيفية (Adaptive Thresholds) |
| تداخل الأسهم | كشف متعدد المقاييس (Multi-scale) |
| غموض التصنيف AON/AOA | تقييم الثقة + مراجعة بشرية |
| أداء الصور الكبيرة | أهرامات الصور ومعالجة مناطق الاهتمام (ROI) |

---

*هذه الوثيقة خارطة (Blueprint) تنفيذ محلل PERT والمسار الحرج — النسخة الإنجليزية الأصل في `ARCHITECTURE.md`.*
