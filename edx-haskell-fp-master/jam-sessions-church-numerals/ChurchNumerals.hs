module ChurchNumerals where

-- type Church = (a->a)->a->a

zero = \s z -> z
one = \s z -> s z
-- two = \s z -> s (s z) by eta reduction
two = \s -> s . s

c2i x = x (+1) 0
-- c2i two = (+1) ((+1) 0) = 2

c2s x = x ('*':) "" -- "*****" = 5
-- c2s two = (\s z -> s . s z) ('*':) ""
--      = ('*':) ('*':"") = '*':'*' = '**'