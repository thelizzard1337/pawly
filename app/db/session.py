from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.core.config import Settings


def make_database(settings: Settings):
    engine = create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_recycle=300,
        connect_args={"timeout": 5, "command_timeout": 10},
    )
    return engine, async_sessionmaker(engine, expire_on_commit=False)
