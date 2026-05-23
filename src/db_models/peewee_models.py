# from peewee import *

# database = SqliteDatabase('just1stock.db')

# class UnknownField(object):
#     def __init__(self, *_, **__): pass

# class BaseModel(Model):
#     class Meta:
#         database = database

# class AuthGroup(BaseModel):
#     name = CharField(unique=True)

#     class Meta:
#         table_name = 'auth_group'

# class DjangoContentType(BaseModel):
#     app_label = CharField()
#     model = CharField()

#     class Meta:
#         table_name = 'django_content_type'
#         indexes = (
#             (('app_label', 'model'), True),
#         )

# class AuthPermission(BaseModel):
#     codename = CharField()
#     content_type = ForeignKeyField(column_name='content_type_id', field='id', model=DjangoContentType)
#     name = CharField()

#     class Meta:
#         table_name = 'auth_permission'
#         indexes = (
#             (('content_type', 'codename'), True),
#         )

# class AuthGroupPermissions(BaseModel):
#     group = ForeignKeyField(column_name='group_id', field='id', model=AuthGroup)
#     permission = ForeignKeyField(column_name='permission_id', field='id', model=AuthPermission)

#     class Meta:
#         table_name = 'auth_group_permissions'
#         indexes = (
#             (('group', 'permission'), True),
#         )

# class AuthUser(BaseModel):
#     date_joined = DateTimeField()
#     email = CharField()
#     first_name = CharField()
#     is_active = BooleanField()
#     is_staff = BooleanField()
#     is_superuser = BooleanField()
#     last_login = DateTimeField(null=True)
#     last_name = CharField()
#     password = CharField()
#     username = CharField(unique=True)

#     class Meta:
#         table_name = 'auth_user'

# class AuthUserGroups(BaseModel):
#     group = ForeignKeyField(column_name='group_id', field='id', model=AuthGroup)
#     user = ForeignKeyField(column_name='user_id', field='id', model=AuthUser)

#     class Meta:
#         table_name = 'auth_user_groups'
#         indexes = (
#             (('user', 'group'), True),
#         )

# class AuthUserUserPermissions(BaseModel):
#     permission = ForeignKeyField(column_name='permission_id', field='id', model=AuthPermission)
#     user = ForeignKeyField(column_name='user_id', field='id', model=AuthUser)

#     class Meta:
#         table_name = 'auth_user_user_permissions'
#         indexes = (
#             (('user', 'permission'), True),
#         )

# class DjangoAdminLog(BaseModel):
#     action_flag = IntegerField()
#     action_time = DateTimeField()
#     change_message = TextField()
#     content_type = ForeignKeyField(column_name='content_type_id', field='id', model=DjangoContentType, null=True)
#     object_id = TextField(null=True)
#     object_repr = CharField()
#     user = ForeignKeyField(column_name='user_id', field='id', model=AuthUser)

#     class Meta:
#         table_name = 'django_admin_log'

# class DjangoMigrations(BaseModel):
#     app = CharField()
#     applied = DateTimeField()
#     name = CharField()

#     class Meta:
#         table_name = 'django_migrations'

# class DjangoSession(BaseModel):
#     expire_date = DateTimeField(index=True)
#     session_data = TextField()
#     session_key = CharField(primary_key=True)

#     class Meta:
#         table_name = 'django_session'

# class MyappActivestocks(BaseModel):
#     stock_id = CharField(primary_key=True)

#     class Meta:
#         table_name = 'myapp_activestocks'

# class MyappPortfolio(BaseModel):
#     amount = FloatField(null=True)
#     date = DateField(primary_key=True)

#     class Meta:
#         table_name = 'myapp_portfolio'

# class MyappPositions(BaseModel):
#     cost = FloatField()
#     entry_bar = IntegerField()
#     entry_date = DateField()
#     entry_price = FloatField()
#     highest = FloatField()
#     shares = FloatField()
#     stock_id = CharField()

#     class Meta:
#         table_name = 'myapp_positions'
#         indexes = (
#             (('stock_id', 'entry_date'), True),
#         )

# class MyappStocksibbuysell(BaseModel):
#     buy = BigIntegerField()
#     date = DateField()
#     name = CharField()
#     sell = BigIntegerField()
#     stock_id = CharField()

#     class Meta:
#         table_name = 'myapp_stocksibbuysell'
#         indexes = (
#             (('stock_id', 'date', 'name'), True),
#         )

# class MyappStocksinfo(BaseModel):
#     group = CharField()
#     market_type = CharField()
#     name = CharField()
#     stock_id = CharField(primary_key=True)

#     class Meta:
#         table_name = 'myapp_stocksinfo'

# class MyappStocksmargin(BaseModel):
#     date = DateField()
#     margin_purchase_buy = FloatField()
#     margin_purchase_cash_repayment = FloatField()
#     margin_purchase_limit = FloatField()
#     margin_purchase_sell = FloatField()
#     margin_purchase_today_balance = FloatField()
#     margin_purchase_yesterday_balance = FloatField()
#     note = CharField()
#     offset_loan_and_short = FloatField()
#     short_sale_buy = FloatField()
#     short_sale_cash_repayment = FloatField()
#     short_sale_limit = FloatField()
#     short_sale_sell = FloatField()
#     short_sale_today_balance = FloatField()
#     short_sale_yesterday_balance = FloatField()
#     stock_id = CharField()

#     class Meta:
#         table_name = 'myapp_stocksmargin'
#         indexes = (
#             (('stock_id', 'date'), True),
#         )

# class MyappStocksupdateflag(BaseModel):
#     date = DateField()
#     flag = IntegerField()
#     note = CharField()
#     stock_id = CharField()

#     class Meta:
#         table_name = 'myapp_stocksupdateflag'
#         indexes = (
#             (('stock_id', 'date', 'note'), True),
#         )

# class MyappTrades(BaseModel):
#     entry_date = DateField()
#     entry_price = FloatField()
#     exit_date = DateField()
#     exit_price = FloatField()
#     pnl = FloatField()
#     return_pct = FloatField()
#     size = FloatField()
#     stock_id = CharField()

#     class Meta:
#         table_name = 'myapp_trades'
#         indexes = (
#             (('stock_id', 'entry_date'), True),
#         )

# class SqliteSequence(BaseModel):
#     name = BareField(null=True)
#     seq = BareField(null=True)

#     class Meta:
#         table_name = 'sqlite_sequence'
#         primary_key = False
