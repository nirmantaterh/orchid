"""
Known-good answers for every task. Used to prove the graders accept a
correct answer (tests/test_eval.py) and by the fake provider for dry runs.
"""
from eval.tasks import TASKS

CODE_REF = {
    "is_palindrome": "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n",
    "fizzbuzz": "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '')\n        out.append(s or str(i))\n    return out\n",
    "second_largest": "def second_largest(nums):\n    u = sorted(set(nums), reverse=True)\n    return u[1] if len(u) > 1 else None\n",
    "flatten": "def flatten(lst):\n    out = []\n    for x in lst:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out\n",
    "word_count": "import re\ndef word_count(text):\n    d = {}\n    for w in re.split(r'[^a-z]+', text.lower()):\n        if w:\n            d[w] = d.get(w, 0) + 1\n    return d\n",
    "merge_intervals": "def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out\n",
    "roman_to_int": "def roman_to_int(s):\n    v = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    total = 0\n    for i, c in enumerate(s):\n        if i + 1 < len(s) and v[c] < v[s[i+1]]:\n            total -= v[c]\n        else:\n            total += v[c]\n    return total\n",
    "is_balanced": "def is_balanced(s):\n    pairs = {')':'(', ']':'[', '}':'{'}\n    st = []\n    for c in s:\n        if c in '([{':\n            st.append(c)\n        elif c in pairs:\n            if not st or st.pop() != pairs[c]:\n                return False\n    return not st\n",
    "fib": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "two_sum": "def two_sum(nums, target):\n    seen = {}\n    for i, x in enumerate(nums):\n        if target - x in seen:\n            return [seen[target - x], i]\n        seen[x] = i\n",
    "run_length_encode": "def run_length_encode(s):\n    out, i = [], 0\n    while i < len(s):\n        j = i\n        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(s[i] + str(j - i))\n        i = j\n    return ''.join(out)\n",
    "rotate_matrix": "def rotate_matrix(m):\n    return [list(r) for r in zip(*m[::-1])]\n",
    "gcd": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
    "dedupe": "def dedupe(lst):\n    seen, out = [], []\n    for x in lst:\n        if x not in seen:\n            seen.append(x)\n            out.append(x)\n    return out\n",
    "binary_search": "def binary_search(arr, x):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == x:\n            return mid\n        if arr[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
    "valid_ipv4": "def valid_ipv4(s):\n    parts = s.split('.')\n    if len(parts) != 4:\n        return False\n    for p in parts:\n        if not p.isdigit() or (len(p) > 1 and p[0] == '0') or int(p) > 255:\n            return False\n    return True\n",
}

FORMAT_REF = {
    "format-01": "- Catches bugs early\n- Makes refactoring safer\n- Documents expected behaviour",
    "format-02": '{"language": "Python", "year": 1991}',
    "format-03": "An API is a set of rules that lets programs talk to each other.",
    "format-04": "NO",
    "format-05": "1, 2, 3, 4, 5",
    "format-06": "WHY SLEEP MATTERS SO MUCH",
    "format-07": "Tidal motion is calm today. A gull flaps past soft sand.",
    "format-08": "C\nGo\nPython\nRust",
}


def reference_answer(task: dict) -> str:
    kind = task["kind"]
    if kind == "number":
        a = task["answer"]
        return f"Working it out...\nANSWER: {round(a, 4) if isinstance(a, float) else a}"
    if kind == "text":
        return f"Reasoning done.\nANSWER: {task['accept'][0]}"
    if kind == "code":
        return f"```python\n{CODE_REF[task['func']]}```"
    if kind == "format":
        return FORMAT_REF[task["id"]]
    if kind == "code_script":
        return f"```python\n{task['reference']}\n```"
    raise ValueError(kind)


ALL_REFERENCES = {t["id"]: reference_answer(t) for t in TASKS}
