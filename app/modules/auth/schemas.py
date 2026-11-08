from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class Login(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email: EmailStr
    password: str = Field(min_length=10, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value):
        return str(value).strip().lower()


class Register(Login):
    display_name: str = Field(min_length=1, max_length=100)


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class CsrfOut(BaseModel):
    csrf_token: str
