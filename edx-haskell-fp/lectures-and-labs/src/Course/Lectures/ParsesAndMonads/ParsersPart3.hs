module Course.Lectures.ParsesAndMonads.ParsersPart3 where

{-
Define grammar:
expr -> term '+' expr | term
term -> factor '*' term | factor
factor -> digit | '(' expr ')'
digit -> '0' | '1' | ... | '9'
-}

{-
Factorize the rules for expr and term
expr -> term ('+' expr | '')
term -> factor ( '*' term | '')
-}


-- Defined functions in the end of Part 2
{-
expr :: Parser Int
expr = do t <- term
          do char '+'
             e <- expr
             return (t + e)
          +++ return t


term :: Parser Int
term = do f <- factor
          do char '*'
             t <- term
             return (f * t)
          +++ return f

factor :: Parser Int
factor = do d <- digit
            return (digitToInt d)
         +++ do char '('
                e <- expr
                char ')'
                return e
-}

{-
Extension to parse subtraction and division:
expr -> term ('+' expr | '-'   expr | '')
term -> factor ('*' term | '/' term | '')
-}
