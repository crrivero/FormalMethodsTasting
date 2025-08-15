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