module Course.Labs.ListComprehensionEx where

-- Factors/Divisors of a number w/ guards
factors :: Int -> [Int]
factors n = [x | x <- [1..n], n `mod` x == 0]

perfects n = [x | x <- [1..n], isPerfect x]
    where isPerfect = \x -> sum (init (factors x)) == x

