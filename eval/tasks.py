"""
Task set for the external evaluation (eval-v2).

Every task has an answer that can be checked without asking an LLM:
  - "number": numeric answer, read from the last "ANSWER:" line
  - "text":   short text answer, read from the last "ANSWER:" line
  - "code":   a Python function, run against hidden test cases
  - "format": an instruction-following check done in plain Python

Categories: math, facts, logic, code, format.
Answers were checked in code (see tests/test_eval.py).
"""

ANSWER_SUFFIX = (
    "\n\nEnd your response with one final line in exactly this form:\n"
    "ANSWER: <your answer>"
)

CODE_SUFFIX = (
    "\n\nReturn the complete function in a single ```python code block. "
    "Do not include example usage or tests."
)


def _num(id_, prompt, answer, tol=1e-6):
    return {"id": id_, "category": "math", "kind": "number",
            "prompt": prompt + ANSWER_SUFFIX, "answer": answer, "tol": tol}


def _text(id_, category, prompt, accept, reject=()):
    # reject: wrong answers that must NOT appear in the ANSWER line
    # (stops "9.11 is larger than 9.9" from passing on "9.9")
    return {"id": id_, "category": category, "kind": "text",
            "prompt": prompt + ANSWER_SUFFIX, "accept": accept, "reject": list(reject)}


def _code(id_, prompt, func, tests):
    return {"id": id_, "category": "code", "kind": "code",
            "prompt": prompt + CODE_SUFFIX, "func": func, "tests": tests}


def _fmt(id_, prompt, check, **params):
    return {"id": id_, "category": "format", "kind": "format",
            "prompt": prompt, "check": check, "params": params}


MATH = [
    _num("math-01", "Pens cost $3 each, or $10 for a pack of 4. What is the lowest total cost, in dollars, to buy exactly 10 pens?", 26),
    _num("math-02", "A train travels 180 km at 60 km/h and then 120 km at 40 km/h. What is its average speed for the whole trip, in km/h?", 50),
    _num("math-03", "What is the sum of all integers from 1 to 200 (inclusive) that are divisible by 3 or by 5?", 9368),
    _num("math-04", "A rectangle has a perimeter of 46 cm and an area of 120 square cm. How long is its longer side, in cm?", 15),
    _num("math-05", "You invest $1000 at 5% interest per year, compounded once a year. How much is it worth after 3 years, in dollars, rounded to the nearest cent?", 1157.63, tol=0.005),
    _num("math-06", "How many minutes are there between 9:47 AM and 2:15 PM on the same day?", 268),
    _num("math-07", "In how many ways can you choose a group of 3 people from 10 people (order does not matter)?", 120),
    _num("math-08", "If 5 machines take 5 minutes to make 5 widgets, how many minutes would 100 machines take to make 100 widgets?", 5),
    _num("math-09", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost, in dollars?", 0.05, tol=1e-9),
    _num("math-10", "What is the remainder when 7^100 is divided by 13?", 9),
    _num("math-11", "A recipe for 4 people uses 300 g of flour. How many grams of flour are needed for 10 people?", 750),
    _num("math-12", "A price of $80 is increased by 25%, and the new price is then reduced by 20%. What is the final price, in dollars?", 80),
    _num("math-13", "What is the sum of the digits of 2^20?", 31),
    _num("math-14", "How many prime numbers are there that are less than 100?", 25),
    _num("math-15", "What is the mean of these numbers: 4, 8, 15, 16, 23, 42?", 18),
    _num("math-16", "What is the median of these numbers: 7, 3, 9, 1, 5, 11?", 6),
    _num("math-17", "Worker A can finish a job alone in 6 hours. Worker B can finish it alone in 3 hours. Working together, how many hours do they take?", 2),
    _num("math-18", "Two fair six-sided dice are rolled. What is the probability that the total is 8? Give the answer as a fraction or a decimal.", 5 / 36, tol=0.0005),
]

FACTS = [
    _text("facts-01", "facts", "What is the chemical symbol for sodium?", ["na"]),
    _text("facts-02", "facts", "What is the capital city of Australia?", ["canberra"]),
    _text("facts-03", "facts", "Who wrote the novel 'Pride and Prejudice'?", ["austen"]),
    _text("facts-04", "facts", "What is the boiling point of water at sea level, in degrees Celsius? Give only the number.", ["100"]),
    _text("facts-05", "facts", "In what year was the paper 'Attention Is All You Need' (which introduced the Transformer) published?", ["2017"]),
    _text("facts-06", "facts", "What is the largest planet in our solar system?", ["jupiter"]),
    _text("facts-07", "facts", "What does the HTTP status code 404 mean? Give its standard two-word name.", ["not found"]),
    _text("facts-08", "facts", "What is the time complexity of binary search on a sorted array, in Big-O notation?", ["o(log n)", "o(logn)", "o(log(n))", "o(log2 n)", "o(log_2 n)"]),
    _text("facts-09", "facts", "How many bits are in one byte?", ["8"]),
    _text("facts-10", "facts", "What is the smallest prime number?", ["2"]),
    _text("facts-11", "facts", "Which chemical element has atomic number 6?", ["carbon"]),
    _text("facts-12", "facts", "What is the default TCP port for HTTPS?", ["443"]),
]

LOGIC = [
    _text("logic-01", "logic", "Alice is older than Bob. Bob is older than Carol. Dave is younger than Carol. Who is the youngest?", ["dave"], reject=["alice", "bob", "carol"]),
    _text("logic-02", "logic", "Today is Wednesday. What day of the week will it be 100 days from today?", ["friday"], reject=["monday", "tuesday", "wednesday", "thursday", "saturday", "sunday"]),
    _text("logic-03", "logic", "All bloops are razzies, and all razzies are lazzies. Are all bloops definitely lazzies? Answer yes or no.", ["yes"], reject=["no"]),
    _text("logic-04", "logic", "A farmer has 17 sheep. All but 9 run away. How many sheep does the farmer have left?", ["9"]),
    _text("logic-05", "logic", "How many times does the letter 'r' appear in the word 'strawberry'?", ["3"]),
    _text("logic-06", "logic", "Sort these words alphabetically: pear, apple, mango, banana, cherry. Which word is third?", ["cherry"]),
    _text("logic-07", "logic", "Write the word 'evaluation' backwards.", ["noitaulave"]),
    _text("logic-08", "logic", "What is the next number in this sequence: 2, 6, 12, 20, 30, ?", ["42"]),
    _text("logic-09", "logic", "An analog clock shows exactly 3:00. What is the smaller angle between the hour hand and the minute hand, in degrees? Give only the number.", ["90"]),
    _text("logic-10", "logic", "Tom has twice as many apples as Jerry. Together they have 18 apples. How many apples does Tom have?", ["12"]),
    _text("logic-11", "logic", "In a race, you overtake the runner in second place. What place are you in now?", ["2", "2nd", "second"], reject=["1", "1st", "first", "3", "3rd", "third"]),
    _text("logic-12", "logic", "Which number is larger: 9.11 or 9.9?", ["9.9"], reject=["9.11"]),
]

CODE = [
    _code("code-01", "Write a Python function `is_palindrome(s: str) -> bool` that returns True if `s` reads the same forwards and backwards, ignoring case and ignoring every character that is not a letter or digit.",
          "is_palindrome",
          [(["A man, a plan, a canal: Panama"], True), (["race a car"], False), ([""], True), (["No 'x' in Nixon"], True), (["ab"], False)]),
    _code("code-02", "Write a Python function `fizzbuzz(n: int) -> list[str]` that returns a list for the numbers 1 to n: 'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for multiples of both, and the number as a string otherwise.",
          "fizzbuzz",
          [([5], ["1", "2", "Fizz", "4", "Buzz"]), ([15], ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]), ([0], [])]),
    _code("code-03", "Write a Python function `second_largest(nums: list[int])` that returns the second largest DISTINCT value in the list, or None if there is no such value.",
          "second_largest",
          [([[3, 1, 4, 1, 5]], 4), ([[5, 5, 5]], None), ([[]], None), ([[2, 7]], 2), ([[-1, -3, -2]], -2)]),
    _code("code-04", "Write a Python function `flatten(lst: list) -> list` that flattens a list nested to any depth, keeping the original order. Only lists count as nesting.",
          "flatten",
          [([[1, [2, [3, [4]]], 5]], [1, 2, 3, 4, 5]), ([[]], []), ([[[[]], [1]]], [1]), ([["a", ["b"]]], ["a", "b"])]),
    _code("code-05", "Write a Python function `word_count(text: str) -> dict[str, int]` that counts words. Lowercase the text, treat any run of characters that are not letters as a separator, and return a dict mapping each word to its count.",
          "word_count",
          [(["Hello, hello world!"], {"hello": 2, "world": 1}), ([""], {}), (["It's a dog's life"], {"it": 1, "s": 2, "a": 1, "dog": 1, "life": 1})]),
    _code("code-06", "Write a Python function `merge_intervals(intervals: list[list[int]]) -> list[list[int]]` that merges overlapping intervals (intervals that touch, like [1,2] and [2,3], also merge) and returns them sorted by start.",
          "merge_intervals",
          [([[[1, 3], [2, 6], [8, 10], [15, 18]]], [[1, 6], [8, 10], [15, 18]]), ([[[1, 4], [4, 5]]], [[1, 5]]), ([[]], []), ([[[5, 6], [1, 2]]], [[1, 2], [5, 6]])]),
    _code("code-07", "Write a Python function `roman_to_int(s: str) -> int` that converts a valid Roman numeral (I, V, X, L, C, D, M, using subtractive notation like IV and CM) to an integer.",
          "roman_to_int",
          [(["III"], 3), (["LVIII"], 58), (["MCMXCIV"], 1994), (["IX"], 9)]),
    _code("code-08", "Write a Python function `is_balanced(s: str) -> bool` that returns True if every bracket in `s` among (), [], {} is correctly matched and nested. Ignore all other characters.",
          "is_balanced",
          [(["([]{})"], True), (["([)]"], False), ([""], True), (["((("], False), (["a(b)c]"], False)]),
    _code("code-09", "Write a Python function `fib(n: int) -> int` that returns the n-th Fibonacci number, where fib(0) = 0 and fib(1) = 1. It must handle n up to 90 quickly.",
          "fib",
          [([0], 0), ([1], 1), ([10], 55), ([90], 2880067194370816120)]),
    _code("code-10", "Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that returns the indices [i, j] with i < j of the two numbers that add up to target. Exactly one solution exists.",
          "two_sum",
          [([[2, 7, 11, 15], 9], [0, 1]), ([[3, 2, 4], 6], [1, 2]), ([[3, 3], 6], [0, 1])]),
    _code("code-11", "Write a Python function `run_length_encode(s: str) -> str` that replaces each run of the same character with the character followed by the run length. Example: 'aaabcc' becomes 'a3b1c2'.",
          "run_length_encode",
          [(["aaabcc"], "a3b1c2"), ([""], ""), (["abc"], "a1b1c1"), (["zzzzzzzzzzzz"], "z12")]),
    _code("code-12", "Write a Python function `rotate_matrix(m: list[list[int]]) -> list[list[int]]` that returns a new matrix rotated 90 degrees clockwise. The input is a square matrix.",
          "rotate_matrix",
          [([[[1, 2], [3, 4]]], [[3, 1], [4, 2]]), ([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [[7, 4, 1], [8, 5, 2], [9, 6, 3]]), ([[[1]]], [[1]])]),
    _code("code-13", "Write a Python function `gcd(a: int, b: int) -> int` that returns the greatest common divisor of two non-negative integers. gcd(0, 0) should return 0. Do not import anything.",
          "gcd",
          [([12, 18], 6), ([17, 5], 1), ([0, 9], 9), ([0, 0], 0)]),
    _code("code-14", "Write a Python function `dedupe(lst: list) -> list` that removes duplicates while keeping the first occurrence of each item in its original order.",
          "dedupe",
          [([[3, 1, 3, 2, 1]], [3, 1, 2]), ([[]], []), ([["b", "a", "b"]], ["b", "a"])]),
    _code("code-15", "Write a Python function `binary_search(arr: list[int], x: int) -> int` that returns the index of x in the sorted list arr, or -1 if x is not present. All values are distinct.",
          "binary_search",
          [([[1, 3, 5, 7, 9], 7], 3), ([[1, 3, 5, 7, 9], 4], -1), ([[], 1], -1), ([[2], 2], 0)]),
    _code("code-16", "Write a Python function `valid_ipv4(s: str) -> bool` that returns True if s is a valid IPv4 address: four decimal numbers from 0 to 255 separated by dots, with no leading zeros (except the number 0 itself) and no other characters.",
          "valid_ipv4",
          [(["192.168.1.1"], True), (["255.255.255.255"], True), (["256.1.1.1"], False), (["01.2.3.4"], False), (["1.2.3"], False), (["1.2.3.a"], False), (["0.0.0.0"], True)]),
]

FORMAT = [
    _fmt("format-01", "List exactly 3 benefits of unit testing as a bulleted list. Start every line with '- '. Output nothing except the 3 lines.",
         "bullets_exact", n=3),
    _fmt("format-02", "Output a JSON object with exactly two keys: \"language\" set to \"Python\", and \"year\" set to the year Python was first released, as an integer. Output only the JSON object, with no code fences and no other text.",
         "json_exact", expected={"language": "Python", "year": 1991}),
    _fmt("format-03", "In one sentence of no more than 15 words, explain what an API is. Output only that sentence.",
         "one_sentence_max_words", max_words=15),
    _fmt("format-04", "Is 91 a prime number? Reply with only the single word YES or NO.",
         "single_word", expected="no"),
    _fmt("format-05", "Write the numbers 1 to 5 separated by commas. Output nothing else.",
         "exact_csv_numbers", expected=[1, 2, 3, 4, 5]),
    _fmt("format-06", "Write a title for an article about sleep. Use ALL CAPITAL LETTERS and at most 6 words. Output only the title on one line.",
         "caps_title", max_words=6),
    _fmt("format-07", "Write exactly two sentences about the ocean without using the letter 'e' anywhere. Output only the two sentences.",
         "no_letter_sentences", letter="e", n=2),
    _fmt("format-08", "List 4 programming languages, one per line, in alphabetical order. Output nothing else.",
         "sorted_lines", n=4),
]

TASKS = MATH + FACTS + LOGIC + CODE + FORMAT
TASKS_BY_ID = {t["id"]: t for t in TASKS}
