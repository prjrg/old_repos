module Course.Lectures.ParsesAndMonads.ParsersPart1 (
Parser,
return,
failure
) where

import Prelude hiding (return)

type Parser a =  String -> [(a,String)]

item :: Parser Char
item = \inp -> case inp of
  [] -> []
  (x:xs) -> [(x,xs)]

failure :: Parser a
failure = \inp -> []

return :: a -> Parser a
return v = \inp -> [(v, inp)]

parse :: Parser a -> String -> [(a, String)]
parse p inp = p inp

(+++) :: Parser a -> Parser a -> Parser a
p +++ q = \inp -> case p inp of
                    [] -> parse q inp
                    [(v, out)] -> [(v, out)]


