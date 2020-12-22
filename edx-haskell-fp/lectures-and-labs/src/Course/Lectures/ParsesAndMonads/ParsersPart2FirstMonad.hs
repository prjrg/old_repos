module Course.Lectures.ParsesAndMonads.ParsersPart2FirstMonad where

--import Course.Lectures.ParsesAndMonads.ParsersPart1 (return, failure, Parser)
import Prelude hiding (return, failure)
import Data.Char
import Control.Monad

-- Temporary definitions
{-
newtype Parser a =  P (String -> [(a,String)])

failure :: Parser a
failure = P (const [])

parse :: Parser a -> String -> [(a, String)]
parse (P p) = p

return v = P (\inp -> [(v, inp)])
instance Monad Parser where
  p >>= f = P (\inp -> case parse p inp of
                          [] -> []
                          [(v,out)] -> parse (f v) out)
-}

{-

p :: Parser (Char, Char)
p =
    do x <- item
       item
       y <- item
       return (x,y)


sat :: (Char -> Bool) -> Parser Char
sat p = do x <- item
           if p x then return x else failure

digit :: Parser Char
digit = sat isDigit

char :: Char -> Parser Char
char x = sat (x ==)

many :: Parser a -> Parser [a]
many p = many1 p +++ return []

many1 :: Parser a -> Parser [a]
many1 p = do v <- p
             vs <- many p
             return (v:vs)

string :: String -> Parser String
string [] = return []
string (x:xs) = do char x
                   string xs
                   return (x:xs)


p1 :: Parser String
p1 = do char '['
        d <- digit
        ds <- many (do char ',' digit)
        char ']'
        return (d:ds)
-}
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

eval :: String -> Int
eval xs = fst (head (parse expr xs))
-}


