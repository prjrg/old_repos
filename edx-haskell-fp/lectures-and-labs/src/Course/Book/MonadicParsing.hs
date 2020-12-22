module Course.Book.MonadicParsing where

-- Standard libraries for applicative functors and characters
import Control.Applicative
import Data.Char

-- What is a parser?
-- A Parser is a program that takes a string of characters as input, and produces some form of tree
-- that makes the syntactic structure of the string explicit.

-- 1. Parsers as functions
-- Naturally, a parser can be seen as a function that takes a string and produces a tree. Hence, given a suitable type
-- Tree of trees, the notion of a parser can be represented as a function of type String -> Tree.
data Tree a = Leaf a | Node a (Tree a) (Tree a)
type Parser1 a = String -> Tree a
-- In general, however, a parser might not consume its entire argument string. For example, a parser for numbers might
-- be applied to a string comprising a number followed by a word.
-- For this reason, we generalise our type for parsers to also return any unconsumed part of the argument string:
type Parser2 a = String -> (Tree a, String)

-- Similarly, a parser might not always succeed. For example, a parser for numbers might be applied to a string
-- comprising a word. To handle this, we further generalise our type for parsers to return a list of results,
-- with the convention that the empty list denotes failure, and a singleton list denotes success:
type Parser3 a = String -> [(Tree a, String)]

-- In the event of not returning a Tree but an integer or something else, we define Parser4 as follows
type Parser4 a = String -> [(a, String)]
-- From the similarity of the Parser type and the State type, we see that a Parser is a kind of State transformer,
-- that can deal with the possibility of failure.

newtype Parser a = P (String -> [(a, String)])

parse :: Parser a -> String -> [(a, String)]
parse (P p) = p

-- First parsing primitive, which fails if the input string is empty, and succeeds with the first character as the
-- result value otherwise:
item :: Parser Char
item = P (\inp -> case inp of
            [] -> []
            (x:xs) -> [(x, xs)]
          )

-- Sequencing Parsers
-- We now make the parser type into an instance of the functor, applicative and monad classes,
-- in order that the do notation can then be used to combine parsers in sequence.
-- NOTE:: It's very similar to State transformers, except we take into account the possibility that a parser may fail.
instance Functor Parser where
    -- fmap :: (a->b)->Parser a-> Parser b
    fmap g p = P (\inp -> case parse p inp of
                  [] -> []
                  [(v, out)] -> [(g v, out)]
                )

instance Applicative Parser where
    --pure :: a -> Parser a
    pure v = P (\inp -> [(v, inp)])

    -- <*> :: Parser (a->b) -> Parser a -> Parser b
    pg <*> px = P (\inp -> case parse pg inp of
                    [] -> []
                    [(g, out)] -> parse (fmap g px) out
                  )

-- A Parser that consumes 3 characters, discards the second, and returns the first and third as a pair can now be
-- in applicative style:
three :: Parser (Char, Char)
three = pure g <*> item <*> item <*> item
        where g x y z = (x, z)

-- Finally making the Parser type into a Monad:
instance Monad Parser where
    -- (>>=) :: Parser a -> (a -> Parser b)-> Parser b
    p >>= f = P (\inp -> case parse p inp of
                  [] -> []
                  [(v, out)] -> parse (f v) out
                )

-- Since, Parser is now a monadic type, we give a different version of the three parser:
three1 :: Parser (Char, Char)
three1 = do x <- item
            item
            z <- item
            return (x, z)

 -- Making Choices
-- The do notation combines parsers in sequence, with the output string from each parser in the sequence
-- becoming the input string for the next. Another natural way of combining parsers is to apply one parser to the input
-- string, and if this fails to then apply another to the same input instead. We now consider how such a choice operator
-- can be defined for parsers.
-- This idea is captured in the class declaration of Control.Aplicative:
-- class Applicative f => Alternative f where
--    empty :: f a
--    (<|>) :: f a -> f a -> fa
-- Laws for alternatives:
--   empty <|> x = x
--   x <|> empty = x
--   x <|> (y <|> z) = (x <|> y) <|> z

-- A motivating example of an Alternative type is the Maybe type, for which empty is given by the failure value Nothing,
-- and <|> returns its first argument if this succeeds, and its second argument otherwise
data Maybe1 a = Nothing1 | Just1 a

-- Implementing Maybe1 as Functor, Applicative and Alternative
instance Functor Maybe1 where
  --fmap :: (a->b)->Maybe1 a->Maybe1 b
  fmap g m = case m of
              Nothing1 -> Nothing1
              Just1 x -> Just1 (g x)

instance Applicative Maybe1 where
  -- pure :: a -> Maybe1 a
  pure = Just1

  -- <*> :: Maybe1 (a->b) -> Maybe1 a -> Maybe1 b
  Nothing1 <*> _ = Nothing1
  Just1 g <*> m = fmap g m

instance Alternative Maybe1 where
  -- empty :: Maybe a
  empty = Nothing1

  -- (<|>) :: Maybe a -> Maybe a -> Maybe a
  Nothing1 <|> my = my
  (Just1 x) <|> _ = Just1 x

-- The instance for the Parser type is a natural extension of this idea, where empty is the parser that always fails,
-- regardless of the input string, and <|> is a choice operator that returns the result of the first parser if it
-- succeeds on the input, and applies the second parser to the same input otherwise:
instance Alternative Parser where
  -- empty :: Parser a
  empty = P (\inp -> [])

  -- <|> :: Parser a -> Parser a -> Parser a
  p <|> q = P (\inp -> case parse p inp of
              [] -> parse q inp
              r  -> r
            )

-- In summary, Alternatives provide a way to handle failure, which means, if a parser fails we consider the second
-- to the input string passed to the first parser as it was.
-- Note that the Control.Monad library provides a class MonadPlus that plays the same role as Alternatives but for
-- Monadic types, with primitives classed mzero and mplus. However, we are using the applicative choice primitives empty
-- and <|> for parsers because of their similarity to the corresponding symbols for grammars.

-- DERIVED PRIMITIVES
-- We have now 3 basic parsers: item consumes a single character if the input string is non-empty, return v always
-- succeeds with the result value v, and empty always fails.
-- In combination with sequencing and choice, these primitives can be used to define a number of other useful parsers.
-- First of all, we define a parser sat p for single characters that satisfy the predicate p:
sat :: (Char->Bool)-> Parser Char
sat p = do x <- item
           if p x then return x else empty

-- Using sat and appropriate predicates from the library Data.Char, we now define parsers for single digits, lower-case
-- letters, upper-case letters, arbitrary letters, alphanumeric characters, and specific characters:
digit :: Parser Char
digit = sat isDigit

lower :: Parser Char
lower = sat isLower

upper :: Parser Char
upper = sat isUpper

letter :: Parser Char
letter = sat isAlpha

alphanum :: Parser Char
alphanum = sat isAlphaNum

char :: Char -> Parser Char
char x = sat (==x)

-- Parser for the string of characters xs, with the string itself return as the result value:
string :: String -> Parser String
string [] = return []
string (x:xs) = do char x
                   string xs
                   return (x:xs)

-- The next parser many and some apply a parser p as many times as possible until it fails, with the result values
-- from each successful application of b being returned in a list. The difference between these 2 repetition primitives
-- is that many permits zero or more applications of p, whereas some requires at least one successfull application.
-- In fact, the many and some are already provided in the Alternative class:
-- class Applicative f => Alternative f where
--    empty :: f a
--    (<|>) :: f a -> f a -> f a
--    many  :: f a -> f [a]
--    some  :: f a -> f [a]
--
--    many x = some x <|> pure []
--    some x = pure (:) <*> x <*> many x   -- many and some are defined using mutual recursion

-- Using many and some, we can now define parsers for identifiers (variable names) comprising a lower-case letter
-- followed by zero or more alphanumeric characters, natural numbers comprising one or more digits and spacing comprising
-- zero or more space, tab, and newline characters:

ident :: Parser String
ident = do x <- lower
           xs <- many alphanum
           return (x:xs)

nat :: Parser Int
nat = do xs <- some digit
         return (read xs)

space :: Parser ()
space = do many (sat isSpace)
           return ()

-- Parser for integer values
int :: Parser Int
int = do char '-'
         n <- nat
         return (-n)
      <|> nat

-- Handling spacing
-- Most real life parsers allow spacing to be freely used around the basic tokens in their input string. To handle such
-- spacing, we define a new primitive that ignores any space before and after applying a parser for a token.
token :: Parser a -> Parser a
token p = do space
             v <- p
             space
             return v

-- Using token, we can now define parsers that ignore spacing around identifiers, natural numbers, integers and special
-- symbols.

identifier :: Parser String
identifier = token ident

natural :: Parser Int
natural = token nat

integer :: Parser Int
integer = token int

symbol :: String -> Parser String
symbol xs = token (string xs)

-- Now, we can use these primitives to define a parser for a non-empty list of natural numbers that ignores spacing
-- around tokens:
nats :: Parser [Int]
nats = do symbol "["
          n <- natural
          ns <- many (do symbol ","; natural)
          symbol "]"
          return (n:ns)

-- Arithmetic Expressions