import ctypes
import sys
from itertools import product
from string import ascii_uppercase
from types import FunctionType
from typing import Union, List, Callable, Tuple, Iterable, Any

#
# Colors
# imported from https://github.com/Radolyn/RadLibrary/tree/master/RadLibrary/Colors
#

#
# Font decorations
#
BOLD = '\x1b[1m'
UNDERLINE = '\x1b[4m'

#
# Text color
#
GREEN = '\x1b[32m'
LIGHT_BLUE = '\x1b[36m'
LIGHT_YELLOW = '\x1b[93m'
LIGHT_RED = '\x1b[91m'

#
# Mixes
#
STEP_DEC = BOLD + LIGHT_YELLOW
RESULT_DEC = UNDERLINE + GREEN
HEADER = STEP_DEC

#
# Color and font reset
#
RESET_COLOR = '\x1b[39m'
RESET_FONT = '\x1b[0m'
RESET = RESET_COLOR + RESET_FONT

#
# Setup environment
# imported from https://github.com/Radolyn/LogManager/tree/master/LogManager.py
#
if sys.platform == "win32":
    kernel32 = ctypes.windll.kernel32
    kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

#
# Types
#
BooleanFunction = Union[FunctionType, Callable[[bool], bool]]
BooleanArguments = List[Tuple[bool]]

#
# Constants
#
TOKEN_AND = '*'
TOKEN_OR = '+'
TOKEN_NOT = '¬'
TOKEN_SHEFFER = '↑'


#
# Wrapper for benchmarking
#
def timeit(f: FunctionType):
    def wrapper(*args, **kwargs):
        from datetime import datetime

        start = datetime.now()
        f(*args, **kwargs)
        stop = datetime.now()

        res = stop - start

        print(f'\n\n{STEP_DEC}Hack done in {RESET}{RESULT_DEC}{res.total_seconds()} seconds{RESET}')

    return wrapper


#
# Helper classes
#
class Table:
    def __init__(self, headers: Iterable[str]):
        self.headers = list(headers)
        self.headers.append('●')

        self.__column_size = len(max(headers, key=lambda x: len(x))) + 2
        self.__column_count = len(self.headers)
        self.__row_size = (((self.__column_size + 1) * self.__column_count) - 1)

    def generate_top(self):
        s = ''

        s += '┌'
        s += '┬'.join(['─' * self.__column_size for _ in self.headers])
        s += '┐'

        s += '\n'

        s += '│'
        s += '│'.join([f'{HEADER}{arg:^{self.__column_size}}{RESET}' for arg in self.headers])
        s += '│'

        s += '\n'

        s += '├'
        s += '┼'.join(['─' * self.__column_size for _ in self.headers])
        s += '┤'

        return s

    def generate_bottom(self):
        s = ''

        s += '├'
        s += '┼'.join(['─' * self.__column_size for _ in self.headers])
        s += '┤'

        s += '\n'

        s += '│'
        s += '│'.join([f'{HEADER}{arg:^{self.__column_size}}{RESET}' for arg in self.headers])
        s += '│'

        s += '\n'

        s += '└'
        s += '┴'.join(['─' * self.__column_size for _ in self.headers])
        s += '┘'
        return s

    def generate_row(self, *args: Any):
        args = [f'{arg:^{self.__column_size}}' for arg in args]

        s = ''

        s += '│'
        s += '│'.join(args)
        s += '│'

        return s


#
# Helper functions
#
def inspect_args(f: BooleanFunction):
    # P. S. signature is slower than __code__ in 4 times.
    return f.__code__.co_argcount, f.__code__.co_varnames


def generate_args(arg_count: int) -> BooleanArguments:
    return list(product(*[[False, True]] * arg_count))


def generate_dnf_arg(arg_names: List[str], *args):
    args = list(args)
    res = []

    for i in range(len(args)):
        arg = args[i]
        if arg:
            res.append(arg_names[i])
        else:
            res.append(TOKEN_NOT + arg_names[i])

    return res


def print_step(step: str, result: str):
    print(f'{STEP_DEC}{step:<17}{RESET}: {RESULT_DEC}{result}{RESET}')


def print_dnf_step(method: str, dnf_expressions: List[List[str]]):
    s = '(' + f') {TOKEN_OR} ('.join([f' {TOKEN_AND} '.join(item) for item in dnf_expressions]).strip() + ')'
    print_step(method, s)


#
# Simplification
#
def get_alter_value(s: str):
    return '0' if s == '1' else '1'


def mul(x, y):
    res = []
    for i in x:
        if len(i) == 2 and i[0] in y:
            return []
        else:
            res.append(i)
    for i in y:
        if i not in res:
            res.append(i)
    return res


def multiply(x, y):
    res = []
    for i in x:
        for j in y:
            tmp = mul(i, j)
            if len(tmp) != 0:
                res.append(tmp)
    return res


def petric(vars, step2):
    letters = {ascii_uppercase[i]: var for i, var in enumerate(vars)}

    res = []
    for i in range(len(step2[0])):
        exp = []
        for j in range(len(step2)):
            if step2[j][i]:
                exp.append(ascii_uppercase[j])

        if len(exp) == 0:
            print('fuck?? todo')
        else:
            res.append(exp)

    while len(res) > 1:
        res[1] = multiply(res[0], res[1])
        res.pop(0)

    final = [min(res[0], key=len)][0]
    super_final = [letters[item] for item in final]

    return super_final


def simplify(arg_names: List[str], dnf_expressions: List[List[str]]):
    bits = []
    for item in dnf_expressions:
        b = ''
        for var in item:
            b += '0' if var.startswith(TOKEN_NOT) else '1'

        bits.append(b)

    minterms = dict()
    for item in bits:
        count = item.count('1')

        if count in minterms:
            minterms[count].append(item)
        else:
            minterms[count] = [item]

    implicants = set()
    prev_terms = []

    while 1:
        if prev_terms == minterms:
            break
        new_terms = []

        values = list(minterms.values())
        l = len(values)

        for i in range(l):
            minterm = values[i]
            replaced = 0

            for left_term in minterm:
                for j in range(l):
                    minterm2 = values[j]
                    if minterm == minterm2 or i - 3 > j:
                        continue

                    for right_term in minterm2:
                        res = None

                        for i in range(len(left_term)):
                            if left_term[i] == right_term[i]:
                                pass
                            elif left_term[i] == get_alter_value(right_term[i]) or left_term[i] == '-' or right_term[
                                i] == '-':
                                if res is None:
                                    res = i
                                else:
                                    res = None
                                    break
                            else:
                                res = None
                                break

                        if res is not None:
                            replaced += 1
                            s = right_term[0:res] + '-' + right_term[res + 1:]
                            new_terms.append(s)

                if replaced == 0:
                    implicants.add(left_term)

        prev_terms = minterms
        minterms = dict()
        for item in new_terms:
            count = item.count('1')

            if count in minterms:
                minterms[count].append(item)
            else:
                minterms[count] = [item]

    vars = []
    for s in implicants:
        impl = []
        for i, ch in enumerate(s):
            if ch == '1':
                impl.append(arg_names[i])
            elif ch == '0':
                impl.append(TOKEN_NOT + arg_names[i])

        vars.append(impl)

    step2 = [[False for _ in dnf_expressions] for _ in vars]

    for i, impl in enumerate(vars):
        for j, orig in enumerate(dnf_expressions):
            if all(arg in orig for arg in impl):
                step2[i][j] = True

    sdnf = petric(vars, step2)
    print_dnf_step('Simplified', sdnf)

    return sdnf


def sheffer(sdnf):
    sheffer_sdnf = [item.copy() for item in sdnf]

    for i in range(len(sheffer_sdnf)):
        item = sheffer_sdnf[i]
        for j in range(len(item)):
            if item[j][0] == TOKEN_NOT:
                s = item[j].replace(TOKEN_NOT, '')
                item[j] = f'({s} {TOKEN_SHEFFER} {s})'

    res = []
    for item in sheffer_sdnf:
        s = f' {TOKEN_SHEFFER} '.join(item)
        res.append(s)

    final = []
    for item in res:
        s = f'({item} {TOKEN_SHEFFER} {item})'
        final.append(s)

    final_res = f' {TOKEN_SHEFFER} '.join(final)

    print_step('Sheffer', final_res)


#
# Entry point
#
@timeit
def hack(f: BooleanFunction):
    arg_count, arg_names = inspect_args(f)
    arg_names = list(arg_names)

    table = Table(arg_names)

    print(table.generate_top())

    dnf_expressions = []
    for arg in generate_args(arg_count):
        res = f(*arg)
        print(table.generate_row(*arg, res))

        if res:
            dnf_expressions.append(generate_dnf_arg(arg_names, *arg))

    print(table.generate_bottom())

    if len(dnf_expressions) == 2 ** arg_count:
        print_step('DNF', '1')
        print()
        print_step('Simplified', '1')
    elif len(dnf_expressions) == 0:
        print_step('DNF', '0')
        print()
        print_step('Simplified', '0')
    else:
        print_dnf_step('DNF', dnf_expressions)
        print()
        res = simplify(arg_names, dnf_expressions)

        sheffer(res)
