module Course.Book.CountdownProblem where

data Op = Add | Sub | Mul | Div

instance Show Op where
  show Add = "+"
  show Sub = "-"
  show Mul = "*"
  show Div = "/"

valid :: Op -> Int -> Int -> Bool
valid Add _ _ = True
valid Sub x y = x > y
valid Mul _ _ = True
valid Div x y = x `mod` y == 0

apply :: Op -> Int -> Int -> Int
apply Add x y = x + y
apply Sub x y = x - y
apply Mul x y = x * y
apply Div x y = x `div` y

data Expr = Val Int | App Op Expr Expr

instance Show Expr where
  show (Val n) = show n
  show (App o l r) = brak l ++ show o ++ brak r
                      where
                        brak (Val n) = show n
                        brak e = "(" ++ show e ++ ")"


values :: Expr -> [Int]
values (Val n) = [n]
values (App _ l r) = values l ++ values r

eval :: Expr -> [Int]
eval (Val n) = [n | n>0]
eval (App o l r) = [apply o x y | x <- eval l, y <- eval r, valid o x y]

-- Combinatorial Functions
-- Construct all possible Lists that satisfy certain properties

subs :: [a] -> [[a]]
subs [] = [[]]
subs (x:xs) = yss ++ map (x:) yss
              where yss = subs xs

interleave :: a -> [a] -> [[a]]
interleave x [] = [[x]]
interleave x (y:ys) = (x:y:ys) : map (y:) (interleave x ys)

-- Permutations of a List
perms :: [a] -> [[a]]
perms [] = [[]]
perms (x:xs) = concatMap (interleave x) (perms xs)

-- All possible sublists in any order
choices :: [a] -> [[a]]
choices = concatMap perms . subs

-- Formalising the Countdown Problem:
-- Evaluate if an expression solution to the problem
solution :: Expr -> [Int] -> Int -> Bool
solution e ns n = elem (values e) (choices ns) && eval e == [n]

-- Finding the Solution using Brute Force
split :: [a] -> [([a], [a])]
split [] = []
split [_] = []
split (x:xs) = ([x], xs) : [(x:ls, rs) | (ls, rs) <- split xs]

exprs :: [Int] -> [Expr]
exprs [] = []
exprs [n] = [Val n]
exprs ns = [e | (ls, rs) <- split ns, l <- exprs ls, r <- exprs rs, e <- combine l r]

combine :: Expr -> Expr -> [Expr]
combine l r = [App o l r | o <- ops]
  where ops = [Add, Sub, Mul, Div]

ops :: [Op]
ops = [Add, Sub, Mul, Div]

-- Find solution to a problem
solutions :: [Int] -> Int -> [Expr]
solutions ns n = [e | ms <- choices ns, e <- exprs ms, eval e == [n]]

-- Combining generation and evaluation to improve time
type Result = (Expr, Int)

results :: [Int] -> [Result]
results [] = []
results [n] = [(Val n, n) | n>0]
results ns = [res | (ls, rs) <- split ns, lx <- results ls, ry <- results rs, res <- combine1 lx ry]

combine1 :: Result -> Result -> [Result]
combine1 (l, x) (r, y) = [(App o l r, apply o x y) | o <- ops, valid1 o x y]

solutions1 :: [Int] -> Int -> [Expr]
solutions1 ns n = [e | ms <- choices ns, (e, m) <- results ms, m == n]

-- Exploiting algebraic properties to reduce computational cost
{-
x+y = y+x
x * y = y * x
x * 1 = x
1 * y = y
x / 1 = x
-}

valid1 :: Op -> Int -> Int -> Bool
valid1 Add x y = x <= y
valid1 Sub x y = x > y
valid1 Mul x y = x /= 1 && y /= 1 && x <= y
valid1 Div x y = y /= 1 && x `mod` y == 0