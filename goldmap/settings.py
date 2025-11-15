from pathlib import Path
import os

# ───────────── المسار الأساسي ─────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ───────────── المفاتيح وأعلام التشغيل ─────────────
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "dev-key-change-me")  # غيّرها في الإنتاج
DEBUG = os.getenv("DJANGO_DEBUG", "true").strip().lower() == "true"

# ALLOWED_HOSTS تقبل قائمة مفصولة بفواصل في متغير البيئة
ALLOWED_HOSTS = [
    h.strip() for h in os.getenv("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if h.strip()
]

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
# ملاحظة: يجب وضع CorsMiddleware قبل CommonMiddleware وقبل CsrfViewMiddleware.
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",

    "corsheaders.middleware.CorsMiddleware",  # ← مهم أن يكون مبكرًا
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
# افتراضيًا SQLite للتطوير. يمكن تبديلها عبر متغيرات البيئة.
if os.getenv("DB_ENGINE"):
    DATABASES = {
        "default": {
            "ENGINE": os.getenv("DB_ENGINE"),
            "NAME": os.getenv("DB_NAME"),
            "USER": os.getenv("DB_USER", ""),
            "PASSWORD": os.getenv("DB_PASSWORD", ""),
            "HOST": os.getenv("DB_HOST", ""),
            "PORT": os.getenv("DB_PORT", ""),
        }
    }
else:
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
STATICFILES_DIRS = [BASE_DIR / "static"]        # أثناء التطوير
STATIC_ROOT = BASE_DIR / "staticfiles"          # للإنتاج: collectstatic

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ───────────── Django REST Framework ─────────────
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "UNAUTHENTICATED_USER": None,
}

# ───────────── CORS/CSRF ─────────────
# في التطوير: اسمح للجميع. في الإنتاج: حدّد النطاقات.
CORS_ALLOW_ALL_ORIGINS = True if DEBUG else False

# إن أردت ضبطها يدويًا (تتجاهل السابق إذا فُعّلت):
# املأ متغير البيئة CORS_ORIGINS بقائمة URLs مفصولة بفواصل.
_CORS_ORIGINS = os.getenv("CORS_ORIGINS")
if _CORS_ORIGINS:
    CORS_ALLOW_ALL_ORIGINS = False
    CORS_ALLOWED_ORIGINS = [o.strip() for o in _CORS_ORIGINS.split(",") if o.strip()]

CORS_ALLOW_CREDENTIALS = True  # يسمح بإرسال الكوكيز عبر CORS عند الحاجة

# CSRF Trusted Origins — من الأفضل ضبطها من البيئة في الإنتاج
CSRF_TRUSTED_ORIGINS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
_CSRF_EXTRA = os.getenv("CSRF_TRUSTED_EXTRA")
if _CSRf_EXTRA := os.getenv("CSRF_TRUSTED_EXTRA"):
    CSRF_TRUSTED_ORIGINS += [u.strip() for u in _CSRf_EXTRA.split(",") if u.strip()]

# ───────────── إعدادات أمان مبسطة (عدّلها في الإنتاج) ─────────────
SESSION_COOKIE_SECURE = not DEBUG and True
CSRF_COOKIE_SECURE = not DEBUG and True
SECURE_SSL_REDIRECT = not DEBUG and bool(int(os.getenv("SECURE_SSL_REDIRECT", "0")))
SECURE_REFERRER_POLICY = "strict-origin-when-cross-origin"
SECURE_HSTS_SECONDS = 0 if DEBUG else int(os.getenv("SECURE_HSTS_SECONDS", "0"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = False if DEBUG else bool(int(os.getenv("SECURE_HSTS_INCLUDE_SUBDOMAINS", "0")))
SECURE_HSTS_PRELOAD = False if DEBUG else bool(int(os.getenv("SECURE_HSTS_PRELOAD", "0")))

# ───────────── لوجينغ مختصر مفيد أثناء التطوير ─────────────
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
