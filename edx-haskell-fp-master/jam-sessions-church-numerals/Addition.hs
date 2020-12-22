module Addition where

import ChurchNumerals

{-
    x' = c2i x
    y' = c2i y
    Goal is to Prove: x' + y' = c2i (add x y) = c2i (x+y)
    Proof :>
        x' + y' = c2i x + c2i y
                = x (+1) 0 + c2i y
                = x (+1) (c2i y)
                = x (+1) (y (+1) 0)
                = (\s z -Z x s (y s z)) (+1) 0 | By lambda calculus beta expansion
                = (add x y) (+1) 0
                = c2i (add x y) | By c2i definition
-}

add x y = \s z -> x s (y s z)

