from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
load_dotenv()


pwd =os.getenv("PWD")

DATABASE_URL = (
    f"mysql+pymysql://root:{pwd}"
    "@localhost:3306/ai_customer_service"
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    bind=engine
)