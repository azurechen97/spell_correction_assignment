letters = "abcdefghijklmnopqrstuvwxyz"
other = " -'"
def distance_1(word):
    total = {word}
    for i in range(0,len(word)):
        l = word[:i]
        r = word[i:]
        alphabet = letters
        if len(word) > 1 and i > 0:
            total.update(set([l[:i-1]+r[0]+l[i-1]+r[1:]]))
            if i < len(word):
                alphabet = letters+other
        total.update(set([l+c+r for c in alphabet]))
        total.update(set([l+r[1:]]))
        total.update(set([l+c+r[1:] for c in alphabet]))
    return total

print(distance_1("apple"))