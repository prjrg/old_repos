module Course.Lectures.HigherOrderFunctionsClasses where

twice :: (a->a)->a->a
twice f x = f (f x)

map1 :: (a->b)->[a]->[b]
map1 f [] = []
map1 f (x:xs) = (f x) : map1 f xs


map2 :: (a->b)->[a]->[b]
map2 f xs = [f x | x <- xs]

filter1 :: (a->Bool)->[a]->[a]
filter1 p xs = [x | x <- xs, p x]

filter2 :: (a->Bool)->[a]->[a]
filter2 p [] = []
filter2 p (x:xs)
    | p x = x : filter p xs
    | otherwise = filter p xs

-- The foldr and foldl higher-order functions
foldrP :: (a->b->b)->b->[a]->b
foldrP f acc [] = acc
foldrP f acc (x:xs) = f x (foldrP f acc xs)

sumP = foldrP (+) 0
productP = foldrP (*) 1
andP = foldrP (&&) True


-- More examples of foldr

lengthP :: [a] -> Int
lengthP = foldrP (\_ n -> 1 + n) 0

reverseP :: [a] -> [a]
reverseP = foldrP (\x xs -> xs ++ [x]) []

-- Function composition
(.) :: (b->c)->(a->b)->(a->c)
f . g = \x -> f (g x)

odd :: Int -> Bool
odd = not Prelude.. even

-- My imp of foldl
foldlP :: (a->b->b)->b->[a]->b
foldlP f acc [] = acc
foldlP f acc (x:xs) = foldlP f (f x acc) xs

allP :: (a -> Bool) -> [a] -> Bool
allP p = foldlP (\x1 x2 -> (p x1) && x2) True

allP1 :: (a -> Bool) -> [a] -> Bool
allP1 p xs = and [p x | x <- xs]

anyP :: (a->Bool)->[a]->Bool
anyP p xs = or [p x | x <- xs]

anyP1 :: (a->Bool)->[a]->Bool
anyP1 p = foldlP (\x1 x2 -> (p x1) || x2) False

-- Takewhile
takeWhileP :: (a->Bool)->[a]->[a]
takeWhileP p [] = []
takeWhileP p (x:xs)
    | p x = x : takeWhileP p xs
    | otherwise = []

dropWhileP :: (a->Bool)->[a]->[a]
dropWhileP p [] = []
dropWhileP p (x:xs)
    | p x = dropWhileP p xs
    | otherwise = x:xs
