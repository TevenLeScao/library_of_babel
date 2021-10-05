import json

if __name__ == "__main__":
    word_list = []
    word_count = 0
    with open("count_1w.txt") as f, open("test_words.json", "w") as g:
        for line in f:
            word = line.split()[0]
            if 4 < len(word) < 15:
                word_list.append(word)
                word_count += 1
            if word_count == 10000:
                break
        json.dump(word_list, g, indent=2, ensure_ascii=False)
