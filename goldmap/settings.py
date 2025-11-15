from pathlib import Path
import os

# لو مركّب python-dotenv حمّله، ولو مو مركّب ما بيكسر المشروع
try:
    from dotenv import load_dotenv
except ImportError:  # احتياط لو المكتبة مو منصبة
    load_dotenv = None

# ───────────── المسار الأساسي ─────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# تحميل ملف .env إن كان موجوداً
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

# متغير الدومين الخارجي من Render (مثلاً goldmap.onrender.com)
RENDER_EXTERNAL_HOSTNAME = os.getenv("RENDER_EXTERNAL_HOSTNAME")

# ───────────── المفاتيح وأعلام التشغيل ─────────────
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-key-change-me")  # غيّرها في الإنتاج

# DJANGO_DEBUG في .env (True/False)
DEBUG = os.getenv("DJANGO_DEBUG", "true").strip().lower() == "true"

# ───────────── ALLOWED_HOSTS مصحّحة ─────────────
# نقرأ من DJANGO_ALLOWED_HOSTS إن وُجد، وإلا نستخدم localhost
_env_hosts = [
    h.strip()
    for h in os.getenv("DJANGO_ALLOWED_HOSTS", "").split(",")
    if h.strip()
]

if _env_hosts:
    ALLOWED_HOSTS = _env_hosts
else:
    ALLOWED_HOSTS = ["127.0.0.1", "localhost"]

# إضافة دومين Render تلقائيًا (goldmap.onrender.com) إن وُجد
if RENDER_EXTERNAL_HOSTNAME and RENDER_EXTERNAL_HOSTNAME not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append(RENDER_EXTERNAL_HOSTNAME)

# ───────────── التطبيقات ─────────────
INSTALLED_APPS = [
    # Django
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # خارجي
    "rest_framework",
    "corsheaders",

    # تطبيقاتك المحلية
    "core",          # ابقه إذا عندك تطبيق core
    # "api",         # فعّله إذا استخدمت تطبيق api للمسارات الخلفية
]

# ───────────── Middleware ─────────────
# ملاحظة: يجب وضع CorsMiddleware قبل CommonMiddleware.
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",  # لخدمة static في الإنتاج

    "django.contrib.sessions.middleware.SessionMiddleware",

    "corsheaders.middleware.CorsMiddleware",       # ← مهم أن يكون مبكرًا
    "django.middleware.common.CommonMiddleware",

    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "goldmap.urls"

# ───────────── القوالب ─────────────
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],        # ضع map.html و globe.html هنا
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "goldmap.wsgi.application"

# ───────────── قاعدة البيانات ─────────────
"""
من .env:

DB_NAME=goldmap_db
DB_USER=goldmap_user
DB_PASSWORD=xxxx
DB_HOST=xxxx
DB_PORT=5432

لو هذه القيم موجودة → يستخدم PostgreSQL (Render)
لو غير موجودة → يستخدم SQLite للتطوير
"""

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

if all([DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT]):
    # إنتاج / Postgres
    DATABASES = {
        "default": {
            "ENGINE": os.getenv("DB_ENGINE", "django.db.backends.postgresql"),
            "NAME": DB_NAME,
            "USER": DB_USER,
            "PASSWORD": DB_PASSWORD,
            "HOST": DB_HOST,
            "PORT": DB_PORT,
        }
    }
else:
    # تطوير محلي / SQLite
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

# ───────────── اللغة والمنطقة الزمنية ─────────────
LANGUAGE_CODE = "ar"           # أو "ar-sa" لو أردت محليّة السعودية
TIME_ZONE = "Asia/Riyadh"
USE_I18N = True
USE_TZ = True

# ───────────── الملفات الثابتة والإعلامية ─────────────
STATIC_URL = "/static/"

# أثناء التطوير يمكن أن يكون لديك مجلد static
_static_dir = BASE_DIR / "static"
if _static_dir.exists():
    STATICFILES_DIRS = [_static_dir]
else:
    STATICFILES_DIRS = []

# في الإنتاج: collectstatic يجمع في staticfiles
STATIC_ROOT = BASE_DIR / "staticfiles"

# Whitenoise لتقديم الملفات الثابتة في الإنتاج
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ───────────── Django REST Framework ─────────────
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
        # لو حاب واجهة Browsable API في التطوير فقط:
        *(
            ["rest_framework.renderers.BrowsableAPIRenderer"]
            if DEBUG
            else []
        ),
    ],
    "UNAUTHENTICATED_USER": None,
}

# ───────────── CORS/CSRF ─────────────
# في التطوير: اسمح للجميع. في الإنتاج: يفضّل ضبطها من متغيرات البيئة.
CORS_ALLOW_ALL_ORIGINS = True if DEBUG else False

# إن أردت ضبطها يدويًا من البيئة:
# CORS_ORIGINS="https://site1.com,https://site2.com"
_cors_origins = os.getenv("CORS_ORIGINS")
if _cors_origins:
    CORS_ALLOW_ALL_ORIGINS = False
    CORS_ALLOWED_ORIGINS = [
        o.strip() for o in _cors_origins.split(",") if o.strip()
    ]

CORS_ALLOW_CREDENTIALS = True  # يسمح بإرسال الكوكيز عبر CORS عند الحاجة

# CSRF Trusted Origins — من الأفضل ضبطها للبيئات الحقيقية
CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

# إضافة https://goldmap.onrender.com تلقائيًا إن وُجد
if RENDER_EXTERNAL_HOSTNAME:
    CSRF_TRUSTED_ORIGINS.append(f"https://{RENDER_EXTERNAL_HOSTNAME}")

# من البيئة (مثلاً: CSRF_TRUSTED_EXTRA="https://goldmap.onrender.com,https://domain2.com")
_csrf_extra = os.getenv("CSRF_TRUSTED_EXTRA")
if _csrf_extra:
    CSRF_TRUSTED_ORIGINS += [
        u.strip() for u in _csrf_extra.split(",") if u.strip()
    ]

# ───────────── إعدادات أمان مبسطة (عدّلها في الإنتاج) ─────────────
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG

# إعادة التوجيه لـ HTTPS في الإنتاج فقط لو فعلتها في .env
SECURE_SSL_REDIRECT = not DEBUG and bool(int(os.getenv("SECURE_SSL_REDIRECT", "0")))

SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"

SECURE_HSTS_SECONDS = 0 if DEBUG else int(os.getenv("SECURE_HSTS_SECONDS", "0"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = (
    False if DEBUG else bool(int(os.getenv("SECURE_HSTS_INCLUDE_SUBDOMAINS", "0")))
)
SECURE_HSTS_PRELOAD = False if DEBUG else bool(int(os.getenv("SECURE_HSTS_PRELOAD", "0")))

# ───────────── Logging مبسط مفيد أثناء التطوير ─────────────
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {"class": "logging.StreamHandler"},
    },
    "root": {
        "handlers": ["console"],
        "level": "DEBUG" if DEBUG else "INFO",
    },
}
