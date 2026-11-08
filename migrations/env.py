import asyncio

from alembic import context
from sqlalchemy import pool
from sqlalchemy.dialects.postgresql.base import ischema_names
from sqlalchemy.ext.asyncio import create_async_engine

from app.core.config import Settings
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.models import Geography, Vector

ischema_names["geography"] = Geography
ischema_names["vector"] = Vector


def include_name(name, type_, parent_names):
    # PostGIS owns this table. Never propose dropping extension-managed objects.
    return not (type_ == "table" and name == "spatial_ref_sys")


def run(connection):
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        compare_type=True,
        include_name=include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


async def online():
    engine = create_async_engine(Settings().database_url, poolclass=pool.NullPool)
    async with engine.connect() as conn:
        # The PostGIS image also installs tiger geocoder tables in a separate schema.
        await conn.exec_driver_sql("SET search_path TO public")
        await conn.commit()
        await conn.run_sync(run)
    await engine.dispose()


if context.is_offline_mode():
    context.configure(
        url=Settings().database_url,
        target_metadata=Base.metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(online())
