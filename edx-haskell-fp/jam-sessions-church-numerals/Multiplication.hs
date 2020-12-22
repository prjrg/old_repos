module Multiplication where

import Addition
import ChurchNumerals

three = \f -> f. f. f
six = \s -> (s . s) . (s . s) . (s . s)

mul x y = \s z -> x ( y s) z

