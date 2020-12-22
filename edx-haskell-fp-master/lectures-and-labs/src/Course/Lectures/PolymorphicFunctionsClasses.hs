module Course.Lectures.PolymorphicFunctionsClasses where

length1 :: [a] -> Int -> Int
length1 [] acc = acc
length1 (x:xs) acc = length1 xs (acc+1)

length2 xs = length1 xs 0

sum1 :: Num a => [a] -> a -> a
sum1 [] acc = acc
sum1 (x:xs) acc = sum1 xs (x+acc)

sum2 xs = sum1 xs 0
