"""Report completed corrections and unstarted remote work without experiments."""
from pathlib import Path
import json
from analysis.current_results_report import rows,md
from campaigns.local_fast_tradeoff import save,sha

REPO=Path(__file__).resolve().parents[2]


def read(path):return json.loads(path.read_text())


def main():
    root=REPO/'code/validation/outputs/referee_round3_20261026';validation=read(REPO/'output/prototype_audit_referee_revision_20261026_validation.json')
    policy=read(REPO/'code/validation/outputs/referee_round2_20261026/execution_policy.json')
    retired=policy.get('mobin_server_retired',False)
    execution_note=('مبین دیگر در اختیار شما نیست و از برنامهٔ اجرا کنار گذاشته شد. هیچ تلاش اتصال یا اجرای دیگری روی آن انجام نمی‌شود. محدودیت قبلیِ ران سنگین روی لپتاپ برقرار است و میزبان جایگزینی تعیین نشده است.' if retired else 'سه اتصال SSH هنگام دریافت banner timeout شدند و تلاش چهارم `No route to host` برگرداند. از این خطا به‌تنهایی نمی‌توان علت را سهمیهٔ ترافیک دانست.')
    next_execution_note=('بستهٔ قبلیِ مبین فقط یک آماده‌سازی آرشیوی است و برای اجرا مجاز نیست. ادامهٔ آزمایش‌های سنگین به تعیین یک میزبان مجاز جدید، یا دستور صریح شما برای اجرای محدود روی لپتاپ، نیاز دارد.' if retired else 'برای ادامهٔ بخش سنگین، اتصال SSH مبین باید دوباره در دسترس باشد.')
    assert validation['visual_review']=='PASS' and sha(REPO/validation['pdf'])==validation['pdf_sha256']
    replay=read(REPO/'code/validation/outputs/referee_final_20261025/public_replay_round3_20261026/status.json')
    assert replay['complete'] and replay['verified_total']==7679
    status=dict(paper=validation,additional_public_replay=dict(complete=True,verified=7679,workers=1,go_soft_memory_mib=256,independent_implementation=False),
        unit_checks=dict(python_passed=18,go_passed=2,new_proofs_generated=0),connection=read(root/'connection_status.json'),
        remote_packet=read(root/'packet_receipt.json'),new_heavy_results=False,scientific_issues_all_resolved=False,
        matched_timings=rows('matched_results'),matched_quality=rows('matched_quality'),confirmation=rows('confirmation_quality'))
    save(REPO/'output/referee_round3_status_20261026.json',status)
    report=f'''# گزارش پاسخ به داور جدید — ۲۶ اکتبر ۲۰۲۶

بخش‌های قابل انجام بدون ران سنگین اصلاح شدند و نسخهٔ جدید مقاله ساخته و همهٔ ۱۴ صفحهٔ آن بررسی بصری شد. هیچ عدد عملکردی جدیدی ساخته یا جایگزین نتیجهٔ قبلی نشده است؛ سلول‌های عددی هر ۱۴ فایل جدول با نسخهٔ قبلی مقایسه و یکسان بودنشان تأیید شد.

**همهٔ ایرادهای علمی داور هنوز حل نشده‌اند.** آزمایش‌های سنگین جدید شروع نشده‌اند. {execution_note} روی لپتاپ هیچ prover یا آزمایش سنگین جدیدی اجرا نشد.

## چه چیزی واقعاً انجام شد؟

- contribution اصلی به یک **certificate mechanism مستقل از representation** بازنویسی شد. PCA و centroid-informed دو پیاده‌سازی همان سازوکارند. ادعای برتری عمومی head حذف شد.
- abstract برای خوانندهٔ عمومی ML بازنویسی و شکل preprocessing → query → refinement → Full fallback اضافه شد.
- جدول‌های تکراری pilot از متن اصلی کنار رفتند؛ نتایج قدیمیِ مرتبط، حساسیت زمان و مدل amortization به پیوست منتقل شدند. مقایسه‌های اصلی زمان، کیفیت و هزینهٔ مراحل باقی ماندند.
- نام basisها و معنی early/mature، Concl. و FB روشن شد. prediction agreement، label-certificate coverage و update-decision coverage جدا تعریف شده‌اند.
- آزمایش امنیتی false decision، leakage عمومی و تعریف محدودِ simulation برای proof transcript نوشته شد. این تعریف و استدلال مشروط جای یک composable security audit نیست.
- بازیابی group moments با probe و خطر گروه تک‌عضوی صریح باقی ماند. مقاله دیگر از خصوصی بودن witness، نتیجهٔ پنهان ماندن aggregate یا بازسازی‌ناپذیری همهٔ رکوردها نمی‌گیرد.
- دلیل استفاده از ZK در برابر TEE به تفاوت اعتماد به host/attestation و امکان بررسی artifact مربوط شد؛ هیچ برتری تجربی بر TEE یا deployment واقعی ادعا نشد.
- منابع اولیهٔ مرتبط با zkCNN، ZEN، ZKML، Mystique، zkLLM، EZKL، Artemis، Libra، Nova و aggregation بررسی و بخش related work تکمیل شد. تعداد citation جای مقایسهٔ تجربی را نمی‌گیرد.
- rankهای 63 و 79 به‌عنوان profile منابعِ انتخاب‌شده در development توضیح داده شدند، نه عدد بهینهٔ نظری. cap=16 نیز انتخاب development باقی ماند.

## بررسی تازهٔ اثبات‌های موجود

خطای wrapper مربوط به `COMMITJL_RESUME_PROOFS` اصلاح شد. سپس **۷٬۶۷۹ proof موجود** در فرآیندهای جدید، فقط با فایل‌های عمومی، با موفقیت بررسی شدند. manifest و proof bytes قبل و بعد از replay کنترل شدند. این بررسی با یک worker و هدف نرم حافظهٔ Go برابر 256 MiB انجام شد و هیچ proof یا کلید جدیدی تولید نکرد.

این replay از همان verifierهای منجمد قبلی استفاده می‌کند؛ پیاده‌سازی امنیتی مستقل، تکرار fit یا آزمایش سرعت جدید نیست. کنترل‌های جاری، advance-bank appendix و timing repeat را پوشش می‌دهد. replay جامع بسته‌های قدیمی‌ترِ بازیابی‌شده از مبین، موضوع جداگانه‌ای است و هنوز کامل نشده است.

## نتایج واقعیِ فعلی

هر cohort چهار candidate متمایز دارد. زمان‌ها دقیقه‌اند؛ Full/method بزرگ‌تر از یک یعنی صرفه‌جویی در هزینهٔ فعالِ همان مقایسه. هزینهٔ fallback تازه در عدد نهایی منظور شده است. این‌ها نتیجهٔ قبلیِ لپتاپ‌اند و با زمان آیندهٔ سرور مخلوط نمی‌شوند.

{md(['Workload','N','Method/basis','Minutes','Full/method','Conclusive','Fallback','Proofs'],status['matched_timings'])}

دو سطر Centroid و PCA هر دو certificate ما را اجرا می‌کنند. در workload سخت MNIST، هر دو چهار regression طبیعی با switching را resolve کردند؛ Centroid دو و PCA سه exact-row correction لازم داشت. Moments-only هیچ‌کدام را بدون fallback حل نکرد. صرفه‌جویی به درخواست و نرخ fallback وابسته است؛ عدد پنج‌برابر برای کل مسیرِ این کنترل‌های جدید نداریم.

{md(['Workload','Method/basis','Mean agreement %','Worst %','Label certified %','Cost error %','Interval width %','Regression coverage'],status['matched_quality'])}

agreement از predictor رتبهٔ 63 است؛ certificate رتبهٔ 79 تصمیم تغییر cost را پشتیبانی می‌کند. اصلاح چند ردیف برای تصمیمِ cost، predictor را عوض نمی‌کند. عرض بازه درصدی از cost مرجع است. صفر بودن خطای certificate در موارد آزموده‌شده، به معنی coverage صددرصد برای هر داده و هر query نیست.

در confirmation موجود با دو seed/split جدید، در N=1024 روش Centroid به 16/16 و PCA به 15/16 تصمیم می‌رسد. یک regression در PCA هنوز Full می‌خواهد. این مطالعه native است؛ زمان proof جدید برای آن ثبت نشده است.

## وضعیت هر ایراد داور

| ایراد | وضعیت واقعی | کار باقی‌مانده |
|---|---|---|
| تناسب با Neurocomputing | contribution و مرز ML دقیق‌تر شد؛ ایراد scope باقی است | تصمیم ژورنال با supervisor؛ تغییر لحن، پیشرفت ML ایجاد نمی‌کند |
| برتری نداشتن learned head | framing اصلاح شد؛ PCA یکی از basisهای خود سازوکار است | برتری عمومی head ادعا نمی‌شود؛ نیاز به ساختن چنین ادعایی نداریم |
| مقیاس، k و تکرار کم | کد و طرح جدید آماده‌اند؛ نتیجهٔ تازه ندارند | مقیاس تا 16,384 رکورد، k=10/20 و حداقل سه repetition روی سرور |
| baseline ضعیف | online affine و exact PLONK control پیاده‌سازی شدند؛ هنوز proof timing ندارند | اجرای matched server controls؛ baseline تخصصی matrix-product همچنان جداگانه لازم است |
| امنیت و adaptive privacy | تعریف و leakage دقیق‌تر شدند؛ حفاظت قوی رفع نشده است | سرویس challenge واقعی، تحلیل تخصصی و تغییر interface اگر aggregate hiding لازم باشد |
| motivation و TEE | workflow و انتخاب اعتماد توضیح داده شد | validation کاربرد یا deployment واقعی هنوز موجود نیست |
| post-hoc selection | development و confirmation تفکیک و طرح جدید پیش از outcome ثبت شد | اجرای fit/splitهای جدید با profile ثابت و نگه داشتن همهٔ نتایج |
| authors/declarations و public artifact | طبق دستور شما به supervisor و review خصوصی واگذار شده | هیچ author یا public URL ساختگی اضافه نشد؛ نسخه هنوز submission-ready نیست |

## آزمایش‌های باقی‌مانده — فعلاً بدون میزبان اجرا

بستهٔ خصوصی `referee_round3_mobin_private_20261026.zip` شامل آماده‌سازی قبلیِ کد و ورودی‌های کوچک است. manifest آن بررسی شده و فایل‌های `.pk`، `.proof` و `.gob` قدیمی داخل بسته نیستند. {next_execution_note}

طرح صفِ قبلی ابتدا proof/tamper smokeها و سپس سه اجرای تازهٔ Full و هر دو basis با چهار worker، affine با مرکزِ همان درخواست و PLONK برای همان relation دقیق را در نظر می‌گرفت. برای میزبان بعدی باید تنظیمات اجرایی دوباره مشخص شوند. همهٔ invocationها و شکست‌ها باید در ledger بمانند. این کنترل مدل fitted موجود را ورودی سرویس می‌گیرد؛ هزینهٔ fit پیش‌بینی و basis کمکی هر دو خارج‌اند، و preprocessing اثباتی حساب می‌شود. عدد fit یک میزبان به زمان میزبان دیگر اضافه نمی‌شود.

PLONK در این کنترل از test SRS ناامن استفاده می‌کند و صریحاً benchmark-only است. این ابزار اندازه‌گیری backend است؛ نه secure deployment و نه جایگزین یک matrix-product prover بهینهٔ تخصصی. نتیجهٔ سرعتی برایش نداریم.

کد confirmation مستقل، چهار seed جدید در هر یک از دو dataset، sweep مقدار k، reference ageهای 1/8 و تمام candidateهای متمایز را پوشش می‌دهد. از sample واقعی 50,000تایی، 3,000 ردیف training و held-outهای جدا می‌سازد. grid شامل ۱۴ ترکیب dataset/k/seed و ۵۶ representation fit است، ولی فقط **هشت split مستقلِ dataset/seed** داریم. برای fit اصلی، Nهای 128/512/1024/4096/16384 ثابت شده‌اند. اگر ردیف واقعی کافی نباشد، runner خطا می‌دهد؛ duplication یا padding نمی‌کند.

۱۸ تست Python و دو تست Go پاس شده‌اند. این تست‌ها admission، عدم اجرای سنگین روی لپتاپ، جدایی training/audit، اطلاعات آینده در affine، staging و حفاظت از replay receipt را بررسی می‌کنند. مسیر کاملِ proof تازهٔ PLONK و grid بزرگ هنوز روی سرور smoke/run نشده است. آماده بودن کد را نتیجهٔ تجربی حساب نکرده‌ایم.

فایل‌های اصلی:

- مقاله: `output/pdf/prototype_audit_referee_revision_20261026.pdf`
- منابع قابل ویرایش: `output/prototype_audit_referee_revision_20261026_sources.zip`
- وضعیت ماشینی و جدول‌ها: `output/referee_round3_status_20261026.json`
- طرح قبل از outcome: `code/validation/outputs/referee_round3_20261026/extended_study_plan.json`
- راهنمای اجرای خصوصی: `output/referee_round3_remote_instructions.md`

داده، کد و بسته‌ها خصوصی باقی ماندند و چیزی publish یا push نشد. {next_execution_note} نتایج و فایل‌های قبلاً بازیابی‌شده از سرور حفظ شده‌اند. مسئلهٔ privacy قوی و تناسب ژورنال صرفاً با اجرای بیشتر حل نمی‌شوند و نباید بسته‌شده گزارش شوند.
'''
    (REPO/'output/referee_round3_report_20261026_fa.md').write_text(report,encoding='utf-8')
    registry_path=REPO/'code/validation/outputs/research_jobs_20261024.json'
    registry=read(registry_path)
    current_state=('Mobin retired by the user; no further connections. Heavy laptop runs held; replacement execution target not assigned.' if retired else 'Remote code/inputs prepared; SSH unreachable, no new heavy run started.')
    registry.update(active_processes=[],instructions='Keep all material private and preserve completed evidence. Heavy laptop runs remain held. Honor the current execution policy; do not connect to retired hosts or resume historical queues. Do not claim all scientific issues resolved.',
        latest_paper=validation['pdf'],latest_paper_validation='output/prototype_audit_referee_revision_20261026_validation.json',latest_paper_canonical_sha256=validation['pdf_sha256'],
        paper_ready=True,paper_ready_for_private_review=True,submission_ready=False,heavy_runs_allowed=policy.get('heavy_runs_allowed',False),remote_heavy_runs_allowed=policy.get('remote_heavy_runs_allowed',False),
        current_status_file='code/validation/outputs/referee_round3_20261026/connection_status.json',current_status='Paper corrected and 7679 current proofs replayed. '+current_state,
        mobin_server_retired=retired,mobin_connection_attempts_allowed=policy.get('mobin_connection_attempts_allowed',not retired),replacement_execution_target=policy.get('replacement_execution_target'),
        additional_public_replay_complete=True,additional_public_replay_verified=7679,new_remote_jobs_started=False)
    save(registry_path,registry)
    print(json.dumps(dict(report='output/referee_round3_report_20261026_fa.md',scientific_issues_all_resolved=False,new_heavy_jobs_started=False)))


if __name__=='__main__':main()
