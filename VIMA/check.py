import pickle

# pklファイルを読み込む
with open('check/action.pkl', 'rb') as file:
    data = pickle.load(file)

# データをtxtファイルに書き込む
with open('output.txt', 'w') as txt_file:
    txt_file.write(str(data))

# データを表示する
print(data)
