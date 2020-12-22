module Course.Lectures.ListComprehensions where

import Data.Char


l0 = [(x,y) | x <- [3,4,5], y <- [4,5]]
l1 = [(x,y) | x <- [1..3], y <- [x..3]]

-- concat a list of lists
concat :: [[a]] -> [a]
concat xss = [x | xs <- xss, x <- xs]

-- concat example
c0 = Main.concat [[1,2,3], [4,5], [6]]

-- list comprehensions w/ guards

even1 :: Integral a => [a] -> [a]
even1 xs = [x | x <- xs, Prelude.even x]

-- Factors/Divisors of a number w/ guards
factors :: Int -> [Int]
factors n = [x | x <- [1..n], n `mod` x == 0]

-- Simple prime checker

prime :: Int -> Bool
prime n = factors n == [1, n]

-- Extensive prime listing function

primes :: Int -> [Int]
primes n = [x | x <- [2..n], prime x]

-- !!! Manipulating general lists
pairs :: [a] -> [(a,a)]
pairs xs = zip xs (tail xs)

sorted :: Ord a => [a] -> Bool
sorted xs = and [x<=y | (x,y) <- pairs xs]

positions :: Eq a => a -> [a] -> [Int]
positions x xs = [i | (y, i) <- zip xs [0..n], y == x]
    where n = length xs - 1

-- Working with Strings

lowers :: String -> Int
lowers xs = length [x | x <- xs,  isLower x]