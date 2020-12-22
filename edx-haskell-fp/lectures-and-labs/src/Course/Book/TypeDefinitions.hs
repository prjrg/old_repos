module Course.Book.TypeDefinitions(Assoc, find) where

-- Defining types
type Pos = (Int, Int)

type Trans = Pos -> Pos

type Pair a = (a, a)

type Assoc k v = [(k,v)]

find :: Eq k => k -> Assoc k v -> v
find k t = head [ v | (k1, v) <- t, k1 == k]

-- Data declarations | Creating new types
data Move = North | South | East | West deriving Show

move :: Move -> Pos -> Pos
move North (x,y) = (x, y+1)
move South (x,y) = (x, y-1)
move East (x, y) = (x+1, y)
move West (x, y) = (x-1, y)

move' :: Move -> Pos -> Pos
move' m (x,y) = case m of
                  North -> (x, y+1)
                  South -> (x, y-1)
                  East -> (x+1, y)
                  West -> (x-1, y)

moves :: [Move] -> Pos -> Pos
moves ms p = foldl (\x m -> move m x) p ms

rev :: Move -> Move
rev North = South
rev South = North
rev East = West
rev West = East

data Shape = Circle Float | Rect Float Float deriving Show

square :: Float -> Shape
square n = Rect n n

area :: Shape -> Float
area (Circle r) = pi * r^2
area (Rect x y) = x * y

data Optional a = Failure | Result a

safediv :: Int -> Int -> Maybe Int
safediv _ 0 = Nothing
safediv m n = Just (m `div` n)

safehead :: [a] -> Maybe a
safehead [] = Nothing
safehead xs = Just (head xs)

-- Newtype declarations
newtype Nat0 = N Int
type Nat1 = Int
data Nat2 = N1 Int

-- Recursive Types
data Nat = Zero | Succ Nat deriving (Eq,Show)

nat2int :: Nat -> Int
nat2int Zero = 0
nat2int (Succ n) = 1 + nat2int n

int2nat :: Int -> Nat
int2nat 0 = Zero
int2nat n = Succ (int2nat (n-1))

-- Tail recursive
loop :: Eq a => (a->a)->(b->a->b)->a->b->a-> b
loop r f a acc v
  | a == v = acc
  | otherwise = loop r f a (f acc v) (r v)

-- Tail recursive nat2intTR
nat2intTR:: Nat -> Int
nat2intTR = loop (\n -> case n of Zero -> Zero; Succ n -> n) (\acc n -> 1 + acc) Zero 0

int2natTR :: Int -> Nat
int2natTR = loop (\n -> n-1) (\acc n -> Succ acc) 0 Zero

-- Using transformations to Int and from Int
addWInt :: Nat -> Nat -> Nat
addWInt m n = int2natTR (nat2intTR m + nat2intTR n)

add :: Nat -> Nat -> Nat
add Zero n = n
add (Succ m) n = Succ (add m n)

-- Defining lists
data List1 a = Nil | Cons a (List1 a) deriving (Eq, Show)

len :: List1 a -> Int
len Nil = 0
len (Cons _ xs) = 1 + len xs

lenTR :: Eq a => List1 a -> Int
lenTR = loop (\l -> case l of Nil -> Nil; Cons _ xs -> xs) (\acc l -> 1 + acc) Nil 0

-- Defining Trees
data Tree a = Leaf a | Node (Tree a) a (Tree a) deriving Show

t :: Tree Int
t = Node (Node (Leaf 1) 3 (Leaf 4)) 5 (Node (Leaf 6) 7 (Leaf 9))

occurs :: Eq a => a -> Tree a -> Bool
occurs x (Leaf y) = x == y
occurs x (Node lt y rt) = y == x || occurs x lt || occurs x rt

flatten :: Tree a -> [a]
flatten (Leaf x) = [x]
flatten (Node l x r) = flatten l ++ [x] ++ flatten r

occursSorted :: Ord a => a -> Tree a -> Bool
occursSorted x (Leaf y) = x == y
occursSorted x (Node l y r)
    | x == y = True
    | x < y = occursSorted x l
    | otherwise = occursSorted x r


-- Many Tree versions
data Tree1 a = Leaf1 a | Node1 (Tree1 a) (Tree1 a)
data Tree2 a = Leaf2 | Node2 (Tree2 a) a (Tree2 a)
data Tree3 a b = Leaf3 a | Node3 (Tree3 a b) b (Tree3 a b)
data Tree4 a = Node4 a [Tree4 a]
