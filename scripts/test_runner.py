import os
import subprocess
import sys


def main():
    # Never migrate or truncate a development DB by an accidental test invocation.
    if os.environ.get("APP_ENV") != "test" or not os.environ.get("DB_NAME", "").endswith("_test"):
        raise SystemExit("Tests require APP_ENV=test and a dedicated *_test database")
    subprocess.run(["alembic", "upgrade", "head"], check=True)
    subprocess.run(["alembic", "check"], check=True)
    subprocess.run(["ruff", "check", "."], check=True)
    subprocess.run(["ruff", "format", "--check", "."], check=True)
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"]))


if __name__ == "__main__":
    main()
