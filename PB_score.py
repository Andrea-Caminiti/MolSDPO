if __name__ == '__main__':
    with open('pb.txt', 'r') as f:
        lines = f.readlines()
    e = 0
    for line in lines:
        e += eval(line[line.index('(') + 1:-2])
    print(e/len(lines))

    