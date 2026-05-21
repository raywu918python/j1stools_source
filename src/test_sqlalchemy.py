# from requests import session
# from sqlalchemy import create_engine, Column, Integer, String, Date
# from sqlalchemy.orm import declarative_base, Session

# engine = create_engine("sqlite:///just1stock.db")

# Base = declarative_base()


# class MyappStocksupdateflag(Base):
#     __tablename__ = "myapp_stocksupdateflag"
#     id = Column(Integer, primary_key=True)
#     date = Column(Date)
#     flag = Column(String)
#     note = Column(String)
#     stock_id = Column(String)


# Base.metadata.create_all(engine)

# with Session(engine) as session:
#     rs = session.query(MyappStocksupdateflag).all()
#     print(len(rs))
