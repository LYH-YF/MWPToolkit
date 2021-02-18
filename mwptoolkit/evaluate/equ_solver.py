from scipy.optimize import fsolve
import sympy  as sym
import re
eq="(200.0/x)-(200.0/(10.0+x))=1.0"
equ=["(","200.0","/","x",")","-","200.0","/","(","10.0","+","x",")","=","1"]
equ=["20","-","10","*","x","+","2","=","12"]
symbols={}
def post_solver(expression):
    st = list()
    res = list()
    symbols={}
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in expression:
        if e in ["("]:
            st.append(e)
        elif e == ")":
            num=0
            c = st.pop()
            num+=c
            while c != "(":
                #res.append(c)
                c = st.pop()
                if c == "/":
                    num = st.pop()/num
                elif c=="*":
                    num = st.pop()*num
                elif c=="-":
                    num = st.pop()-num
                elif c=="+":
                    num = st.pop()+num
                else:
                    pass
            st.append(num)
        elif e[0].isdigit():
            st.append(eval(e))
        elif e.isalpha():
            st.append(sym.symbols(e))
        elif e in priority:
            num=0
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                if e == "/":
                    num = st.pop()/num
                elif e=="*":
                    num = st.pop()*num
                elif e=="-":
                    num = st.pop()-num
                elif e=="+":
                    num = st.pop()+num
            st.append(e)
    num=st.pop()
    while len(st) > 0:
        c=st.pop()
        if c == "/":
            num = st.pop()/num
        elif c=="*":
            num = st.pop()*num
        elif c=="-":
            num = st.pop()-num
        elif c=="+":
            num = st.pop()+num
        else:
            pass
    return num
def from_infix_to_postfix(expression):
    r"""postfix for expression
    """
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res
def compute_postfix_expression(post_fix):
        st = list()
        symbols=[]
        operators = ["+", "-", "^", "*", "/"]
        for p in post_fix:
            if p not in operators:
                pos = re.search("\d+\(", p)
                if pos:
                    st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
                elif p[-1] == "%":
                        st.append(float(p[:-1]) / 100)
                elif p.isalpha():
                    x=sym.symbols(p)
                    st.append(x)
                    symbols.append(x)
                else:
                    st.append(eval(p))
            elif p == "+" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a + b)
            elif p == "*" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(a * b)
            elif p == "/" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if a == 0:
                    return None,None
                st.append(b / a)
            elif p == "-" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                st.append(b - a)
            elif p == "^" and len(st) > 1:
                a = st.pop()
                b = st.pop()
                if float(b) != 2.0 and float(b) != 3.0:
                    return None,None
                st.append(a ** b)
            else:
                return None,None
        if len(st) == 1:
            return st.pop(),symbols
        return None,None
for symbol in equ:
    try:
        num=eval(symbol)
        symbols[symbol]=num
    except:
        symbols[symbol]=sym.symbols(symbol)
eq_idx=equ.index("=")
equ1,equ2=equ[:eq_idx],equ[eq_idx+1:]
equ1=from_infix_to_postfix(equ1)
equ2=from_infix_to_postfix(equ2)
s1,symbol1=compute_postfix_expression(equ1)
s2,symbol2=compute_postfix_expression(equ2)
symbol3=symbol1+symbol2
f=sym.Eq(s1,s2)
y=sym.solve(f,symbol3)
a=post_solver(equ[:eq_idx])
x=sym.symbols("x")
print(x)
