with open("record.txt", 'w', encoding = 'utf-8') as f:
    idx = f.read
    print(idx, type(idx)!=int)
    f.write(str(idx + 1))