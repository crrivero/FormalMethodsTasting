from z3 import *
import re
import IPython.display

html_to_latex_dict = {
  r"&not;": r" \neg ",
  r"&or;": r" \lor ",
  r"&and;": r" \land ",
  r"&rArr;": r" \implies ",
  r"&lt;": r" < ",
  r"&gt;": r" > ",
  r"&le;": r" \leq ",
  r"&ge;": r" \geq ",
  r"&ne;": r" \not= ",
  r"&middot;": r" \cdot ",
  r"<sub>u</sub>": r"_u",
}
html_to_latex_dict_pattern = re.compile("|".join(map(re.escape, html_to_latex_dict.keys())))

def HTMLtoLaTeX( s ):
  s2 = re.sub(r"If\((.*?), 1, 0\)", r'\1', s)
  return html_to_latex_dict_pattern.sub(lambda m: html_to_latex_dict[m.group(0)], s2)

def showSolver( solver ):
  set_pp_option("html_mode", True)
  for x in solver.assertions():
    IPython.display.display( IPython.display.Math( ( HTMLtoLaTeX( str(x) ) ) ) )
  set_pp_option("html_mode", False)

# Code to enumerate over all models in a Z3 solver
# Source: https://theory.stanford.edu/%7Enikolaj/programmingz3.html and https://stackoverflow.com/questions/11867611/z3py-checking-all-solutions-for-equation/
# s = solver, initial_terms = set of variables
def all_smt(s, initial_terms):
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))
    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))
    def all_smt_rec(terms):
        if sat == s.check():
           m = s.model()
           yield m
           for i in range(len(terms)):
               s.push()
               block_term(s, m, terms[i])
               for j in range(i):
                   fix_term(s, m, terms[j])
               yield from all_smt_rec(terms[i:])
               s.pop()
    yield from all_smt_rec(list(initial_terms))

def list_all_solutions( s, initial_terms ):
  return list( all_smt(s, initial_terms) )
