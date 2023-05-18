import string
from itertools import permutations
import random

class Counter:
    def __init__(self):
        self.__value = 0

    def inc(self):
        self.__value += 1

    def get(self):
        return self.__value

class Solution:
    def __init__(self, table : dict(), enc : dict(), enc_freq1 : dict(), enc_freq2 : dict(), eng : dict(), eng_freq1 : dict(), eng_freq2 : dict()):
        self.__table = table
        self.enc = enc
        self.eng = eng
        self.eng_freq1 = eng_freq1 
        self.eng_freq2 = eng_freq2
        self.enc_freq1 = enc_freq1
        self.enc_freq2 = enc_freq2
        self.__fit = float('-inf')

        ##w1 * self.__eval_word(enc=enc, english=eng)
    def get_fit(self, counter):
        did_not_calc = bool(self.__fit == float('-inf'))
        if did_not_calc:
            counter.inc()
            self.__fit = self.__eval_word(enc=self.enc, english=self.eng) + self.__eval_freq(enc_freq=self.enc_freq1, eng_freq=self.eng_freq1, epsilon=0.05) + self.__eval_freq(enc_freq=self.enc_freq2, eng_freq=self.eng_freq2, epsilon=0.005)
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
    
    def get_perm(self):
        perm = "{"
        for let in string.ascii_lowercase:
            perm += let + " : " + self.__table[let]
            if let == 'z':
                perm += "}"
            else:
                perm += ", "
        return perm
    
    def mutation(self):
        c1, c2 = random.sample(string.ascii_lowercase, 2)
        val1 = self.__table[c1]
        val2 = self.__table[c2]
        t = self.__table.copy()
        t[c1] = val2
        t[c2] = val1
        return t
            
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


def to_dict(text : str):
    ans = dict()
    alphabet = string.ascii_lowercase
    for i in range(0, 26):
        ans[alphabet[i]] = text[i]        
    return ans


def select_next(options : list, counter):
    best = None
    score = float('-inf')
    for s in options:
        temp = s.get_fit(counter)
        if temp > score:
            score = temp
            best = s
    return best


def pair_solutions(solutions):
    ans = []
    random.shuffle(solutions)
    size = len(solutions)
    last = int(size / 2)
    for i in range(0, last):
        if 2 * i + 1 >= len(solutions):
            ans.append((solutions[2 * i], solutions[2 * i]))
        else:
            ans.append((solutions[2 * i], solutions[2 * i + 1]))
    return ans


## random crossover
def crossover(solutions, enc, enc_freq1, enc_freq2, eng, eng_freq1, eng_freq2):
    alphabet = list(string.ascii_lowercase)
    stay_odds = 0.35
    mu_odds = 0.01
    next_gen = []
    for p in solutions:
        children = random.randint(1, 2)
        sol1 = p[0]
        sol2 = p[1]
        for i in range(0, children):

            random.shuffle(alphabet)
            random_division = random.randint(1, 25)
            letters1 = alphabet[:random_division]
            letters2 = alphabet[random_division:]

            table1 = dict()
            for let in letters1:
                table1[let] = sol1.convert(let)
            for let in letters2:
                table1[let] = sol2.convert(let)
            off1 = Solution(table=table1, enc=enc, eng=eng, enc_freq1=enc_freq1, enc_freq2=enc_freq2, eng_freq1=eng_freq1, eng_freq2=eng_freq2)
          
            next_gen.append(off1)
            if random.random() <= mu_odds:
                
                next_gen.append(Solution(table=off1.mutation(), enc=enc, eng=eng, enc_freq1=enc_freq1, enc_freq2=enc_freq2, eng_freq1=eng_freq1, eng_freq2=eng_freq2))
            if (random_division != len(alphabet) / 2):
                table2 = dict()
                for let in letters1:
                    table2[let] = sol2.convert(let)
                for let in letters2:
                    table2[let] = sol1.convert(let)
                off2 = Solution(table=table2, enc=enc, eng=eng, enc_freq1=enc_freq1, enc_freq2=enc_freq2, eng_freq1=eng_freq1, eng_freq2=eng_freq2)
                
                next_gen.append(off2)
                if random.random() <= mu_odds:
                    
                    next_gen.append(Solution(table=off2.mutation(), enc=enc, eng=eng, enc_freq1=enc_freq1, enc_freq2=enc_freq2, eng_freq1=eng_freq1, eng_freq2=eng_freq2))

        if random.random() <= stay_odds:
            next_gen.append(sol1)
        if random.random() <= stay_odds:
            next_gen.append(sol2)

    return next_gen


def genetic(enc, enc_freq1, enc_freq2, english, eng_freq1, eng_freq2, num_permutations=2000, max_gen=800, max_con=15):
    alphabet = string.ascii_lowercase
    c = Counter()
    gen = 0
    con = 0
    best_score = float('-inf')
    best_sol = None

    limited_permutations = set() 

    # Generate limited permutations
    while len(limited_permutations) < num_permutations:
        permutation = random.sample(alphabet, len(alphabet))
        limited_permutations.add(''.join(permutation))

    solutions = [Solution(table=to_dict(x), enc=enc, eng=english, enc_freq1=enc_freq1, enc_freq2=enc_freq2, eng_freq1=eng_freq1, eng_freq2=eng_freq2) for x in limited_permutations]
    slice_size = 0

    while gen < max_gen and con < max_con and len(solutions) > slice_size:
        if len(solutions) > 10000:
            slice_size = 3
        else:
            slice_size = 2

        random.shuffle(solutions)
        end = int(0.15 * len(solutions))
        elite = solutions[:end]
        solutions = solutions[end:]
        solutions = [select_next(sub_solution, c) for sub_solution in [solutions[i:i+slice_size] for i in range(0, len(solutions), slice_size)]]
        if len(solutions) % 2 != 0 and len(solutions) > 1:
            random.shuffle(solutions)
            s1 = solutions.pop()
            s2 = solutions.pop()
            solutions.append(select_next([s1, s2], c))
        
        pairs = pair_solutions(solutions)
        
        next_gen = crossover(solutions=pairs, enc=enc, eng=english, enc_freq1=enc_freq1, enc_freq2=enc_freq2, eng_freq1=eng_freq1, eng_freq2=eng_freq2)
        solutions = next_gen + elite
        
        next_sol = select_next(solutions, c)
        
        if next_sol is not None and next_sol.get_fit(c) > best_score:
            best_score = next_sol.get_fit(c)
            best_sol = next_sol
            con = 0
        else:
            con += 1
        gen += 1

    print("Best fit score: " + str(best_score))
    print("Generation: " + str(gen))
    print("Fitness counter: " + str(c.get()))
    return best_sol

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
        if line == '' or line.isspace():
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


    ans = genetic(enc=enc_set, english=english, enc_freq1=freq_single, enc_freq2=freq_pair, eng_freq1=freq1, eng_freq2=freq2)

    with open("perm.txt", 'w') as file:
        file.write(ans.get_perm())
    
    with open("plain.txt", 'w') as file:
        file.write(ans.convert(enc))

if __name__ == "__main__":
    main()
