from sqlalchemy.ext.asyncio import AsyncSession

from app.api.errors import fail
from app.db.models import User, UserContact
from app.modules.users.schemas import ProfileOut, ProfilePatch, UserOut


async def profile(db: AsyncSession, user: User) -> ProfileOut:
    contact = await db.get(UserContact, user.id)
    return ProfileOut(
        **UserOut.model_validate(user).model_dump(),
        phone=contact.phone if contact else None,
        preferred_channel=contact.preferred_channel if contact else None,
    )


async def update_profile(db: AsyncSession, user: User, data: ProfilePatch):
    if "display_name" in data.model_fields_set:
        if data.display_name is None:
            fail(422, "display_name_required")
        user.display_name = data.display_name
    fields = data.model_dump(exclude_unset=True, exclude={"display_name"})
    if fields:
        # Lock the user so simultaneous first contact updates cannot conflict.
        from sqlalchemy import select

        await db.scalar(select(User).where(User.id == user.id).with_for_update())
        contact = await db.get(UserContact, user.id)
        if contact is None:
            contact = UserContact(user_id=user.id)
            db.add(contact)
        for key, value in fields.items():
            setattr(contact, key, value)
    await db.commit()
    return await profile(db, user)
