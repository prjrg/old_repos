module Course.Lectures.Recursion where

factorial :: Int -> Int
factorial n = product [1..n]

factorial1 :: Int -> Int
factorial1 0 = 1
factorial1 n = n * factorial (n-1)

product1 :: [Int] -> Int
product1 [] = 1
product1 (n:ns) = n * product1 ns

length1 :: [a] -> Int
length1 [] = 0
length1 (_:xs) = 1 + length1 xs

reverse1 :: [a] -> [a]
reverse1 [] = []
reverse1 (x:xs) = reverse1 xs ++ [x]

zip1 :: [a] -> [b] -> [(a,b)]
zip1 [] _ = []
zip1 _ [] = []
zip1 (x:xs) (y:ys) = (x,y) : zip1 xs ys

drop1 :: Int -> [a] -> [a]
drop1 0 xs = xs
drop1 _ [] = []
drop1 n (_:xs) = drop1 (n-1) xs

(+++) :: [a] -> [a] -> [a]
[] +++ ys = ys
(x:xs) +++ ys = x : (xs +++ ys)

-- Quicksort algorithm

quicksort :: Ord a => [a] -> [a]
quicksort [] = []
quicksort (x:xs) = quicksort smaller ++ [x] ++ quicksort bigger
    where
        smaller = [y | y <- xs, y <= x]
        bigger = [y | y <- xs, y > x]

replicate1 :: Int -> a -> [a]
replicate1 0 _ = []
replicate1 n x = x:(replicate1 (n-1) x)

(!!!) :: [a] -> Int -> a
(x:_) !!! 0 = x
(x:xs) !!! n = xs !!! (n-1)

elem1 :: Eq a => a -> [a] -> Bool
elem1 x [] = False
elem1 x (y:_) | x == y = True
elem1 x (_:ys) = elem1 x ys



