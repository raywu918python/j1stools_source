from db_models.peewee_models import MyappActivestocks


def query():
    d = MyappActivestocks.select()
    print("**********")
    for row in d:
        print(row.stock_id)


# query()
