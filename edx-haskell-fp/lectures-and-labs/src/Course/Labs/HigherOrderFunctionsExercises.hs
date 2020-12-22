module Course.Labs.HigherOrderFunctionsExercises where

filterP p xs = [x | x <- xs, p x]

allP p xs = foldl (&&) True (map p xs)

anyP p xs = foldr (\x acc -> (p x) || acc) False xs

takeWhileP :: (a->Bool) -> [a] -> [a]
takeWhileP _ [] = []
takeWhileP p (x:xs)
    | p x = x : takeWhileP p xs
    | otherwise = []

dropWhileP :: (a->Bool)->[a]->[a]
dropWhileP _ [] = []
dropWhileP p (x:xs)
    | p x = dropWhile p xs
    | otherwise = x : xs

mapP :: (a->b) -> [a]->[b]
mapP f = foldl (\xs x -> xs ++ [f x]) []

filterP1 p = foldl (\xs x -> if p x then xs ++ [x] else xs) []
filterP2 p = foldr (\x xs -> if p x then x : xs else xs) []

-- Convert a list of decimal values representation of an integer to its decimal value
dec2int :: [Integer]-> Integer
dec2int = foldl (\x y -> 10*x + y) 0


--Invalid implementation
--sumsqreven = compose [sum, map (^2), filter even]

compose :: [a -> a] -> (a->a)
compose = foldr (.) id

--Currying function
curryP :: ((a, b) -> c)->a->b->c
curryP f = \x y -> f (x,y)

--Uncurrying function
uncurryP :: (a->b->c)->(a,b)->c
uncurryP f = \(x,y) -> f x y

-- Unfold function
unfold :: (b -> Bool) -> (b->a) -> (b->b) -> b -> [a]
unfold p h t x
    | p x = []
    | otherwise = h x : unfold p h t (t x)

type Bit = Int

int2bin :: Int -> [Bit]
int2bin 0 = []
int2bin n = n `mod` 2 : int2bin (n `div` 2)

int2binP = unfold (== 0) (`mod` 2) (`div` 2)

-- Splits a list of Bits into sizeable 8 bit lists each
chop8 :: [Bit] -> [[Bit]]
chop8 [] = []
chop8 bits = take 8 bits : chop8 (drop 8 bits)

chop8P = unfold null (take 8) (drop 8)

-- Define map through the unfold function
mapFromUnfold f = unfold null (f . head) tail

-- Implementation of prelude iterate using unfold
iterateP f = unfold (const False) id f