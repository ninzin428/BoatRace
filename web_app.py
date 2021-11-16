"""WEBからデータを取得する
以下のようなエラーがでたらpip install --upgrade protobuf
「module 'google.protobuf.descriptor' has no attribute '_internal_create_key'」

tailscaleを使うことで携帯からでも見れる
https://tailscale.com/


"""
import time

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from datetime import date
import config as cnf


def main():
    """
    """

    st.write('競艇データ分析')
    #sample2()
    jcds = select_jcd()
    hist_order(jcds)
    hist_return(jcds)

def select_jcd():
    """
    """
    selected_items = st.multiselect('Select Race No.', cnf.JCDS.keys())
    if (len(selected_items) == 0):
        selected_items = list(cnf.JCDS.keys())

    selected_item_names = [cnf.JCDS[x] for x in selected_items]
    st.write(f'Selected: {selected_items}')
    st.write(f'Selected: {selected_item_names}')

    return selected_items



def hist_order(jcds):
    """
    """
    df = pd.read_csv(cnf.RESULTLIST_PATH_ORDER_CONCUT)
    df = df.query('着順1_レーサー == 着順1_レーサー')
    df = df.query('レース場コード == @jcds')
    df = df[['レース場コード', '日付', 'レース', '着順1_艇番']]
    df['レース'] = df['レース'].str.replace('R', '')
    df = df.astype(float).astype(int)

    plot_hist(df, column='着順1_艇番', bins=6, x_min=1, x_max=6)

def hist_return(jcds):
    """
    """
    df = pd.read_csv(cnf.RESULTLIST_PATH_RETURN_CONCUT)
    df = df.rename(columns={'2連勝単式組合わせ': '組合わせ_2連勝単式'})
    df = df.rename(columns={'3連勝単式組合わせ': '組合わせ_3連勝単式'})
    df = df.rename(columns={'2連勝単式払戻金': '払戻金_2連勝単式'})
    df = df.rename(columns={'3連勝単式払戻金': '払戻金_3連勝単式'})
    df = df.query('レース場コード == @jcds')
    df = df.query(r'組合わせ_2連勝単式.str.match("^[1-6]-[1-6]$")', engine='python').copy()
    df = df[['レース場コード', '日付', 'レース', '払戻金_2連勝単式', '払戻金_3連勝単式', '組合わせ_2連勝単式', '組合わせ_3連勝単式']]
    df['レース'] = df['レース'].str.replace('R', '')
    df['払戻金_3連勝単式'] = df['払戻金_3連勝単式'].str.replace('¥', '')
    df['払戻金_3連勝単式'] = df['払戻金_3連勝単式'].str.replace(',', '')
    df['払戻金_3連勝単式'] = df['払戻金_3連勝単式'].astype(float).astype(int)
    df['払戻金_3連勝単式'] = df['払戻金_3連勝単式'] / 100
    df['払戻金_3連勝単式'] = df['払戻金_3連勝単式'].astype(int)
    df['払戻金_3連勝単式'] = df['払戻金_3連勝単式'] * 100
    df['払戻金_2連勝単式'] = df['払戻金_2連勝単式'].str.replace('¥', '')
    df['払戻金_2連勝単式'] = df['払戻金_2連勝単式'].str.replace(',', '')
    df['払戻金_2連勝単式'] = df['払戻金_2連勝単式'].astype(float).astype(int)

    plot_hist(df, column='払戻金_3連勝単式', bins=100, x_min=0, x_max=10000)
    plot_hist(df, column='払戻金_2連勝単式', bins=100, x_min=0, x_max=1000)


def plot_hist(df: pd.DataFrame, column: str, bins: int, x_min: int, x_max: int):
    """
    """
    # 描画領域を用意する
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.hist(df[column].values, bins=bins, range=(x_min, x_max))
    ax.set_title(f'ヒストグラム {column}')
    ax.set_xlabel(f'{column} bins: {bins} min: {x_min} max: {x_max}')
    ax.set_ylabel('N')

    # Matplotlib の Figure を指定して可視化する
    st.pyplot(fig)
    plt.close(fig)








def sample1():
    """
    https://note.com/navitime_tech/n/ned827292df6f
    """
    show_df()
    show_map()
    show_widget()

def show_df():
    """
    # 初めての Streamlit
    データフレームを表として出力できます:
    """

    df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })

    df

def show_map():
    """
    # 地図を描画
    """

    map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [35.68109, 139.76719],
    columns=['lat', 'lon'])

    st.map(map_data)    

def show_widget():
    """
    # ウィジェットの例
    """

    if st.checkbox("チェックボックス"):
        st.write("チェックが入りました。")

    selection = st.selectbox("セレクトボックス", ["1", "2", "3"])
    st.write(f"{selection} を選択")

    """
    ## プログレスバーとボタン
    """


    if st.button("ダウンロード"):
        text = st.empty()
        bar = st.progress(0)

        for i in range(100):
            text.text(f"ダウンロード中 {i + 1}/100")
            bar.progress(i + 1)
            time.sleep(0.01)    

def sample2():
    """
    https://blog.amedama.jp/entry/streamlit-tutorial
    """
    #show_text()
    #show_placeholder()
    #show_animation()
    #show_graph()
    #show_graph_amination()
    #show_matplotlib()
    #show_matplotlib_animation()
    #show_dataframe()
    #show_image()
    # リロードしても同じ結果が得られる
    #df = cached_data()
    #st.dataframe(df)

    #bottun()
    #bottun2()
    #checkbox()
    #radio()
    #selectbox()
    #multiselect()
    #slider()
    #slider_range()
    #slider_date()
    #date_input()
    #time_input()
    #text_input()
    #text_area()
    #number_input()
    #number_input2()
    #file_uploader()
    #color_picker()
    #validation()
    #column()
    #container()
    #placeholder_container()
    #expander()
    #sidebar()
    #help()
    select_app()

def show_text():
    """
    """
    # タイトル
    st.title('Application title')
    # ヘッダ
    st.header('Header')
    # 純粋なテキスト
    st.text('Some text')
    # サブレベルヘッダ
    st.subheader('Sub header')
    # マークダウンテキスト
    st.markdown('**Markdown is available **')
    # LaTeX テキスト
    st.latex(r'\bar{X} = \frac{1}{N} \sum_{n=1}^{N} x_i')
    # コードスニペット
    st.code('print(\'Hello, World!\')')
    # エラーメッセージ
    st.error('Error message')
    # 警告メッセージ
    st.warning('Warning message')
    # 情報メッセージ
    st.info('Information message')
    # 成功メッセージ
    st.success('Success message')
    # 例外の出力
    st.exception(Exception('Oops!'))
    # 辞書の出力
    d = {
        'foo': 'bar',
        'users': [
            'alice',
            'bob',
        ],
    }
    st.json(d)

def show_placeholder():
    """
    """
    # プレースホルダーを用意する
    placeholder1 = st.empty()
    # プレースホルダーに文字列を書き込む
    placeholder1.write('Hello, World')

    placeholder2 = st.empty()
    # コンテキストマネージャとして使えば出力先をプレースホルダーにできる
    with placeholder2:
        # 複数回書き込むと上書きされる
        st.write(1)
        st.write(2)
        st.write(3)  # この場合は最後に書き込んだものだけ見える

def show_animation():
    """
    """
    status_area = st.empty()

    # カウントダウン
    count_down_sec = 5
    for i in range(count_down_sec):
        # プレースホルダーに残り秒数を書き込む
        status_area.write(f'{count_down_sec - i} sec left')
        # スリープ処理を入れる
        time.sleep(1)

    # 完了したときの表示
    status_area.write('Done!')
    # 風船飛ばす
    st.balloons()

def show_graph():
    """
    """
    # ランダムな値でデータフレームを初期化する
    data = {
        'x': np.random.random(20),
        'y': np.random.random(20) - 0.5,
        'z': np.random.random(20) - 1.0,
    }
    df = pd.DataFrame(data)
    # 折れ線グラフ
    st.subheader('Line Chart')
    st.line_chart(df)
    # エリアチャート
    st.subheader('Area Chart')
    st.area_chart(df)
    # バーチャート
    st.subheader('Bar Chart')
    st.bar_chart(df)

def show_graph_amination():
    """
    """
    # 折れ線グラフ (初期状態)
    x = np.random.random(size=(10, 2))
    line_chart = st.line_chart(x)

    for i in range(10):
        # 折れ線グラフに 0.5 秒間隔で 10 回データを追加する
        additional_data = np.random.random(size=(5, 2))
        line_chart.add_rows(additional_data)
        time.sleep(0.5)

def show_matplotlib():
    """
    """
    # 描画領域を用意する
    fig = plt.figure()
    ax = fig.add_subplot()
    # ランダムな値をヒストグラムとしてプロットする
    x = np.random.normal(loc=.0, scale=1., size=(100,))
    ax.hist(x, bins=20)
    # Matplotlib の Figure を指定して可視化する
    st.pyplot(fig)

def show_matplotlib_animation():
    """
    """
    # グラフを書き出すためのプレースホルダを用意する
    plot_area = st.empty()
    fig = plt.figure()
    ax = fig.add_subplot()
    x = np.random.normal(loc=.0, scale=1., size=(100,))
    ax.plot(x)
    # プレースホルダにグラフを書き込む
    plot_area.pyplot(fig)

    # 折れ線グラフに 0.5 秒間隔で 10 回データを追加する
    for i in range(10):
        # グラフを消去する
        ax.clear()
        # データを追加する
        additional_data = np.random.normal(loc=.0, scale=1., size=(10,))
        x = np.concatenate([x, additional_data])
        # グラフを描画し直す
        ax.plot(x)
        # プレースホルダに書き出す
        plot_area.pyplot(fig)
        time.sleep(0.5)

def show_dataframe():
    """
    """
    # Pandas のデータフレームを可視化してみる
    data = {
        # ランダムな値で初期化する
        'x': np.random.random(20),
        'y': np.random.random(20),
    }
    df = pd.DataFrame(data)
    # データフレームを書き出す
    st.dataframe(df)
    # st.write(df)  でも良い
    # スクロールバーを使わず一度に表示したいとき
    st.table(df)

def show_image():
    """
    """
    x = np.random.random(size=(400, 400, 3))
    # NumPy 配列をカラー画像として可視化する
    st.image(x)

# 関数の出力をキャッシュする
@st.cache
def cached_data():
    data = {
        'x': np.random.random(20),
        'y': np.random.random(20),
    }
    df = pd.DataFrame(data)
    return df

def bottun():
    """
    """
    # データフレームを書き出す
    data = np.random.randn(20, 3)
    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
    st.dataframe(df)
    # リロードボタン
    st.button('Reload')

def bottun2():
    """
    """
    if st.button('Top button'):
        # 最後の試行で上のボタンがクリックされた
        st.write('Clicked')
    else:
        # クリックされなかった
        st.write('Not clicked')

    if st.button('Bottom button'):
        # 最後の試行で下のボタンがクリックされた
        st.write('Clicked')
    else:
        # クリックされなかった
        st.write('Not clicked')

def checkbox():
    """
    """
    # チェックボックスにチェックが入っているかで処理を分岐する
    if st.checkbox('Show'):
        # チェックが入っているときはデータフレームを書き出す
        data = np.random.randn(20, 3)
        df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        st.dataframe(df)

def radio():
    """
    """
    selected_item = st.radio('Which do you like?',
                             ['Dog', 'Cat'])
    if selected_item == 'Dog':
        st.write('Wan wan')
    else:
        st.write('Nya- nya-')

def selectbox():
    """
    """
    selected_item = st.selectbox('Which do you like?',
                                 ['Dog', 'Cat'])
    st.write(f'Selected: {selected_item}')

def multiselect():
    """
    """
    selected_items = st.multiselect('What are your favorite characters?',
                                    ['Miho Nishizumi',
                                     'Saori Takebe',
                                     'Hana Isuzu',
                                     'Yukari Akiyama',
                                     'Mako Reizen',
                                     ])
    st.write(f'Selected: {selected_items}')

def slider():
    """
    """
    age = st.slider(label='Your age',
                    min_value=0,
                    max_value=130,
                    value=30,
                    )
    st.write(f'Selected: {age}')

def slider_range():
    """
    """
    min_value, max_value = st.slider(label='Range selected',
                                     min_value=0,
                                     max_value=100,
                                     value=(40, 60),
                                     )
    st.write(f'Selected: {min_value} ~ {max_value}')

def slider_date():
    """
    """
    birthday = st.slider('When is your birthday?',
                         min_value=date(1900, 1, 1),
                         max_value=date.today(),
                         value=date(2000, 1, 1),
                         format='YYYY-MM-DD',
                         )
    st.write('Birthday: ', birthday)

def date_input():
    """
    """
    birthday = st.date_input('When is your birthday?',
                             min_value=date(1900, 1, 1),
                             max_value=date.today(),
                             value=date(2000, 1, 1),
                             )
    st.write('Birthday: ', birthday)

def time_input():
    """
    """
    time = st.time_input(label='Your input:')
    st.write('input: ', time)

def text_input():
    """
    """
    text = st.text_input(label='Message', value='Hello, World!')
    st.write('input: ', text)

def text_area():
    """
    """
    text = st.text_area(label='Multi-line message', value='Hello, World!')
    st.write('input: ', text)

def number_input():
    """
    """
    n = st.number_input(label='What is your favorite number?',
                        value=42,
                        )
    st.write('input: ', n)

def number_input2():
    """
    """
    n = st.number_input(label='What is your favorite number?',
                        value=3.14,
                        )
    st.write('input: ', n)

def file_uploader():
    """
    """
    f = st.file_uploader(label='Upload file:')
    st.write('input: ', f)

    if f is not None:
        # XXX: 信頼できないファイルは安易に評価しないこと
        data = f.getvalue()
        text = data.decode('utf-8')
        st.write('contents: ', text)

def color_picker():
    """
    """
    c = st.color_picker(label='Select color:')
    st.write('input: ', c)

def validation():
    """
    """
    name = st.text_input(label='your name:')

    # バリデーション処理
    if len(name) < 1:
        st.warning('Please input your name')
        # 条件を満たないときは処理を停止する
        st.stop()

    st.write('Hello,', name, '!')

def column():
    """
    """
    # カラムを追加する
    col1, col2, col3 = st.columns(3)

    # コンテキストマネージャとして使う
    with col1:
        st.header('col1')

    with col2:
        st.header('col2')

    with col3:
        st.header('col3')

    # カラムに直接書き込むこともできる
    col1.write('This is column 1')
    col2.write('This is column 2')
    col3.write('This is column 3')

def container():
    """
    """
    # コンテナを追加する
    container = st.container()

    # コンテキストマネージャとして使うことで出力先になる
    with container:
        st.write('This is inside the container')
    # これはコンテナの外への書き込み
    st.write('This is outside the container')

    # コンテナに直接書き込むこともできる
    container = st.container()
    container.write('1')
    st.write('2')
    # 出力順は後だがレイアウト的にはこちらが先に現れる
    container.write('3')

def placeholder_container():
    """
    """
    placeholder = st.empty()
    # プレースホルダにコンテナを追加する
    container = placeholder.container()
    # コンテナにカラムを追加する
    col1, col2 = container.columns(2)
    # それぞれのカラムに書き込む
    with col1:
        st.write('Hello, World')
    with col2:
        st.write('Konnichiwa, Sekai')

def expander():
    """
    """
    with st.expander('See details'):
        st.write('Hidden item')

def sidebar():
    """
    """
    # サイドバーにリロードボタンをつける
    st.sidebar.button('Reload')
    # サイドバーにデータフレームを書き込む
    data = np.random.randn(20, 3)
    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
    st.sidebar.dataframe(df)

def help():
    """
    """
    st.help(pd.DataFrame)

def render_gup():
    """GuP のアプリケーションを処理する関数"""
    character_and_quotes = {
        'Miho Nishizumi': 'パンツァーフォー',
        'Saori Takebe': 'やだもー',
        'Hana Isuzu': '私この試合絶対勝ちたいです',
        'Yukari Akiyama': '最高だぜ！',
        'Mako Reizen': '以上だ',
    }
    selected_items = st.multiselect('What are your favorite characters?',
                                    list(character_and_quotes.keys()))
    for selected_item in selected_items:
        st.write(character_and_quotes[selected_item])


def render_aim_for_the_top():
    """トップ！のアプリケーションを処理する関数"""
    selected_item = st.selectbox('Which do you like more in the series?',
                                 [1, 2])
    if selected_item == 1:
        st.write('me too!')
    else:
        st.write('2 mo ii yo ne =)')


def select_app():
    # アプリケーション名と対応する関数のマッピング
    apps = {
        '-': None,
        'GIRLS und PANZER': render_gup,
        'Aim for the Top! GunBuster': render_aim_for_the_top,
    }
    selected_app_name = st.sidebar.selectbox(label='apps',
                                             options=list(apps.keys()))

    if selected_app_name == '-':
        st.info('Please select the app in sidebar')
        st.stop()

    # 選択されたアプリケーションを処理する関数を呼び出す
    render_func = apps[selected_app_name]
    render_func()








if __name__ == '__main__':
    main()
