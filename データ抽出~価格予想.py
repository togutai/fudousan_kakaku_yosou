from retry import retry
import requests
from bs4 import BeautifulSoup
import pandas as pd 
import datetime
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor as rf
from sklearn.metrics import accuracy_score
import category_encoders as ce
import os

#前処理参考URL:https://ishitonton.hatenablog.com/entry/2019/02/24/184253

base_url = 'https://suumo.jp/jj/bukken/ichiran/JJ012FC001/?ar=030&bs=011&cn=9999999&cnb=0&ekTjCd=&ekTjNm=&kb=1&kt=9999999&mb=0&mt=9999999&sc=13101&sc=13102&sc=13103&sc=13104&sc=13105&sc=13113&sc=13106&sc=13107&sc=13108&sc=13118&sc=13121&sc=13122&sc=13123&sc=13109&sc=13110&sc=13111&sc=13112&sc=13114&sc=13115&sc=13120&sc=13116&sc=13117&sc=13119&sc=13201&sc=13202&sc=13203&sc=13204&sc=13205&sc=13206&sc=13207&sc=13208&sc=13209&sc=13210&sc=13211&sc=13212&sc=13213&sc=13214&sc=13215&sc=13218&sc=13219&sc=13220&sc=13221&sc=13222&sc=13223&sc=13224&sc=13225&sc=13227&sc=13228&sc=13229&ta=13&tj=0&po=0&pj=1&pc=50&pn={}'
max_page = 497
currentDateTime = datetime.datetime.now()
date = currentDateTime.date()
year = date.strftime("%Y")
build_information = {
    'URL': [],
    '物件名':[],
    '販売価格':[],
    '所在地':[],
    '沿線・駅':[],
    '専有面積':[],
    '間取り':[],
    'バルコニー':[],
    '築年月':[]
}


def main():
    
    #ページ毎のデータをdataframe化
    df = get_data(max_page)
    
    #各カラムの整形
    #一度簡易的に整形するコードを描いてみて修正していく
    columns_shape(df)
    
    #欠損値を平均値で代入する
    df = df.fillna(df.mean())
    
    #重複値の削除
    df.drop_duplicates(subset=df.columns.values, inplace=True)
    df.drop_duplicates(subset=["URL", "駅"], inplace=True)#URLが被っている場合でも違う駅からのアクセスの場合は違うデータと想定する
    df.drop_duplicates(subset=["物件名", "販売価格", "専有面積", "間取り"], inplace=True)
    df.drop_duplicates(subset=["販売価格", "専有面積", "間取り"], inplace=True)#物件名に記号等（★とか）が入っていた場合にマッチしない場合を想定

    #標準偏差と正規分布
    std = np.std(df.販売価格)
    df = df.query('-3*@std < 販売価格 < 3*@std')

    #カテゴリ変数のlabel-encoding
    list_cols = ['間取り', '沿線', '駅', '市・区'] #対象列指定
    encoder =  ce.OrdinalEncoder(cols=list_cols, drop_invariant=True)
    df = encoder.fit_transform(df)

    #訓練、テストデータの作成
    x = df[['専有面積', '間取り', 'バルコニー', '築年月', '沿線', '駅', '徒歩', '市・区']]
    y = df['販売価格']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    #訓練および評価
    model = rf()
    model.fit(x_train, y_train)
    model.score(X=x_train, y=y_train)

    #学習済みモデルを使用しての予測
    df['予測販売価格'] = model.predict(x)

    #販売価格と予想販売価格の乖離値を出し降順で並び替え
    df['乖離'] = df['予測販売価格'] - df['販売価格']
    df = df.sort_values('乖離', ascending=False)

    df.to_csv(f'/Users/toguchitaichi/desktop/{os.path.basename(__file__)}.csv')
    

        
@retry(tries=3, delay=10, backoff=2)
def get_html(url: str):
    print(url)
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')
    return soup

#ページ毎の取得データをdataframeとして返す
def get_data(max_page: int):
    
    for page_number in range(1, max_page):
        
        print(f'{page_number}取得中')
        url = base_url.format(page_number)
        soup = get_html(url)
        all_data_in_one_page = soup.findAll(class_='property_unit-content')

        for one_data in all_data_in_one_page:
            build_information['URL'].append(one_data.find('h2', class_='property_unit-title').a['href'])
            number = 1
            for item in one_data.find_all(class_='dottable-line'):
                if number==1:
                    build_information[item.dl.dt.text].append(item.dl.dd.text)
                elif number==2:
                    build_information[item.dl.dt.text].append(item.dl.dd.span.text)
                elif number==3:
                    for e in item.find_all('dl'):
                        build_information[e.dt.text].append(e.dd.text)
                elif 3<number<6:
                    for q in item.find_all('td'):
                        build_information[q.dt.text].append(q.dd.text)
                number += 1  
                
    df = pd.DataFrame(build_information)
    return df
    

#販売価格の整形
def sales_price_shaping(value):
    price = re.findall(r"[0-9～.]+", value)

    if '億' in value:
        if len(price)==1:
            value = value.replace('億円', '0000')
            return int(value)
        elif len(price)>=2:
            price_format = '{:0>4}'.format(price[1])
            price = price[0] + price_format
            return int(price)
    else:
        if not len(price)==1 and price[1][0]=="～":
            #価格範囲がある物件の平均値（3000~4000 -> 3500）
            range_average = int(sum([int(re.sub('[～]','', q)) for q in price]) / 2)
            return int(range_average)
        else:
            return int(price[0])
    
#専有面積の整形
def occupied_area_shaping(value):
    menseki = [float(item) for item in re.findall(r'[0-9.]+[0-9]+', value)]
    if len(menseki) > 1:
        return sum(menseki) / 2
    else:
        return menseki[0]



#築年月の整形
def building_age_shaping(value):
    age = re.search(r'[0-9]+', value)
    building_age = int(year) - int(age.group())
    return building_age

    
#バルコニーの整形
def balcony_shaping(value):
    if not value=='-':
        result = re.search(r'[0-9.]+', value)
        return float(result.group())
    else:
        return None
    
#間取りの整形
def floor_plan_shaping(value):
    floor_plan = re.split('[+]',value)[0]
    if 'ワンルーム' in floor_plan:
        floor_plan = '1R'
        return floor_plan
    return floor_plan

#徒歩カラムの整形
def walk_time_count(value):
    num = re.search(r'[0-9]+', value)
    return num.group()

def location_shaping(value):
    location_shaping = re.search(r'(.*市|.*区)', value)
    return location.group()[3:]


#各カラムの整形
def columns_shape(df):
    
    columns = {
        '販売価格': sales_price_shaping,
        '所在地': location_shaping,
        '専有面積': occupied_area_shaping,
        '築年月': building_age_shaping,
        'バルコニー': balcony_shaping,
        '間取り': floor_plan_shaping,
    }
    
    for column_key, column_value in columns.items():
        if column_key == '所在地':
            df['市・区'] = df[f'{column_key}'].apply(column_value)
        else:
            df[f'{column_key}'] = df[f'{column_key}'].apply(column_value)
    
    #沿線・駅カラムを3つのカラムに分解
    df['沿線'] = df['沿線・駅'].apply(lambda value: re.split('[「」]',value)[0])
    df['駅'] = df['沿線・駅'].apply(lambda value: re.split('[「」]',value)[1])
    df['徒歩'] = df['沿線・駅'].apply(lambda value: re.split('[「」]',value)[2])
    df['徒歩'] = df['徒歩'].apply(walk_time_count)


if __name__ == '__main__':
    df = main()

