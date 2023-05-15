import string

class Solution:
    def __init__(self, table : dict(), enc : dict(), enc_freq1 : dict(), enc_freq2 : dict(), eng : dict(), eng_freq1 : dict(), eng_freq2 : dict(), w1=1, w2=1, w3=1):
        self.__table = table
        self.__fit = w1 * self.__eval_word(enc=enc, english=eng) + w2 * self.__eval_freq(enc_freq=enc_freq1, eng_freq=eng_freq1) + w3 * self.__eval_freq(enc_freq=enc_freq2, eng_freq=eng_freq2)


    def get_fit(self):
        return self.__fit
    
    def convert(self, word : str):
        ans = ""
        for letter in word:
            if can_ignore(letter):
                ans += letter
            else:
                ans += self.__table[letter]
        return ans

    ## giva a score for word match
    def __eval_word(self, english, enc):
        found = 0
        overall = len(enc)
        for word in enc:
            dec = self.convert(word)
            if dec in english:
                found += 1

        if overall == 0:
            return 0
    
        return found / overall

    ## giva a score for the frequency match
    def __eval_freq(self, enc_freq, eng_freq, epsilon=0):
        count = 0
        inside = 0
        for key in enc_freq:
            dec_k = self.convert(key)
            if abs(eng_freq[dec_k] - enc_freq[key]) <= epsilon:
                ##print(dec_k + " : " + key + " -> " + str(eng_freq[dec_k]))
                inside += 1
            count += 1

        if count > 0:
            return inside / count
        return 0
        
## chack if it is a character we need to ignore
def can_ignore(c):
    return bool(c == "." or c == "," or c == ";" or c == "\n" or c == " ")

## get frequency for each single character
def get_single_freq(enc):
    freq = dict()
    count = 0

    ## init every letter to 0
    for l in string.ascii_lowercase:
        freq[l] = 0
    
    ## count letters
    for letter in enc:
        if not can_ignore(letter):
            freq[letter] += 1
            count += 1

    if count > 0:
        for key in freq:
            freq[key] /= count
    
    return freq

## get frequency for each character pair
def get_pairs_freq(enc):
    freq = dict()
    count = 0
    previous = ""

    ## init all pairs to 0
    for l1 in string.ascii_lowercase:
        for l2 in string.ascii_lowercase:
            freq[l1 + l2] = 0

    ## count pairs
    for letter in enc:
        if not can_ignore(letter):
            if previous == "":
                previous = letter
            else:
                freq[previous + letter] += 1
                count += 1
        else:
            previous = ""

    if count > 0:
        for key in freq:
            freq[key] /= count
    
    return freq


def genetic(enc, enc_freq1, enc_freq2, english, eng_freq1, eng_freq2):
    table = dict()
    for l in string.ascii_lowercase:
        table[l] = l
    s = Solution(table=table, enc=enc, eng=english, enc_freq1=enc_freq1, enc_freq2=enc_freq2, eng_freq1=eng_freq1, eng_freq2=eng_freq2)
    print(s.get_fit())


def convert_to_set(text):
    ans = set()
    for word in text:
        word = word.strip().strip(".").strip(",").strip(";")
        if len(word) > 0:
            ans.add(word.lower())

    return ans


def convert_freq_file(lines):
    ans = dict()
    for line in lines:
        if line == '\t' or line == '':
            break
        temp = line.split("\t")
        ans[temp[-1].lower()] = float(temp[0])
    return ans

def main():
    english = set(line.strip() for line in open('dict.txt'))
    english.remove("")


    enc_txt = open("enc.txt", 'r')
    enc = enc_txt.read()
    enc_txt.close()

    freq_single = get_single_freq(enc)
    freq_pair = get_pairs_freq(enc)
    enc_set = convert_to_set(enc.split(" "))


    letter_freq_txt = open("Letter_Freq.txt", 'r')
    lf1 = letter_freq_txt.read().split("\n")
    letter_freq_txt.close()
    freq1 = convert_freq_file(lf1)

    letter2_freq_txt = open("Letter2_Freq.txt", 'r')
    lf2 = letter2_freq_txt.read().split("\n")
    letter2_freq_txt.close()
    freq2 = convert_freq_file(lf2)


    genetic(enc=enc_set, english=english, enc_freq1=freq_single, enc_freq2=freq_pair, eng_freq1=freq1, eng_freq2=freq2)


if __name__ == "__main__":
    main()