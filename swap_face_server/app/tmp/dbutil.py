import mysql.connector as pymysql
import time
import os
import pandas as pd
from decouple import config

class mysql:
    con = None
    def __init__(self):
        DBHOST = os.environ.get("DB_HOST",config("DB_HOST"))
        DBUSER = os.environ.get("DB_USER",config("DB_USER"))
        PASSWORD = os.environ.get("DB_PASS",config("DB_PASS"))
        DB = os.environ.get("DB_NAME",config("DB_NAME"))
        DB_PORT = os.environ.get("DB_PORT",int(config("DB_PORT")))
        try:
            # 连接数据库
            self.con = pymysql.connect(host=DBHOST, port=DB_PORT, user=DBUSER,
                                          password=PASSWORD, database=DB, charset='utf8')
            print(self.con.connection_id)
        except pymysql.Error as e:
            print(e)

    def get_cursor(self):
        try:
            if not self.con.open:
                self.con.ping(reconnect=True)
            cursor = self.con.cursor()
        except Exception as e:
            self.con.reconnect(attempts=2,delay=1)
            cursor = self.con.cursor()
        cursor.execute("SET GLOBAL max_allowed_packet=512*1024*1024")  # Set to 256MB
        return cursor

    def query(self,sql="select unit,word,translation from cyber_words_temp limit 2"):
        #cursor = self.con.cursor()
        cursor = self.get_cursor()
        # 利用字符串方式查询
        cursor.execute(sql)
        # 取出字段名称集合
        columns = cursor.column_names
        # 取出全部数据
        result = cursor.fetchall()
        #print('数据表字段名称：{0}'.format(columns))
        #print('查询结果：{0}'.format(result))
        df = pd.DataFrame(list(result), columns=columns)
        #print(df)
        # example of df to json
        """
        result = df.to_json(orient="records")
        parsed = json.loads(result)
        json.dumps(parsed, indent=4)
        """
        # example of json to df
        """
        from pandas.io.json import json_normalize
        df = pd.DataFrame.from_dict(json_normalize(<json_data>), orient='columns')
        """

        cursor.close()
        return df
    """
    insert_sql = "insert into learn_words(resource,unit_name,word,translation,audio_en) values('oxford',%s,%s,%s,%s)"
    data = (unit,word,translation,audio_en)
    """
    def insert(self,sql,data):
        #cursor = self.con.cursor()
        cursor = self.get_cursor()
        if type(data)== list:
            cursor.executemany(sql, data)
        if type(data) == tuple or type(data) == dict:
            cursor.execute(sql, data)
        if None == data:
            cursor.execute(sql)
        # 提交
        self.con.commit()
        # 关闭
        cursor.close()
    def update(self,sql,data):
        #cursor = self.con.cursor()
        cursor = self.get_cursor()
        if type(data) == tuple or type(data) == dict:
            cursor.execute(sql, data)
        if None == data:
            cursor.execute(sql)
        # 提交
        self.con.commit()
        # 关闭
        cursor.close()
    def delete(self,sql,data):
        #cursor = self.con.cursor()
        cursor = self.get_cursor()
        if type(data) == tuple or type(data) == dict:
            cursor.execute(sql, data)
        if None == data:
            cursor.execute(sql)
        # 提交
        self.con.commit()
        # 关闭
        cursor.close()

# if __name__ == "__main__":
#     sql = mysql()
#     sql.query()
#     data = ('Tom3', 22)
#     print(type(data)== tuple)