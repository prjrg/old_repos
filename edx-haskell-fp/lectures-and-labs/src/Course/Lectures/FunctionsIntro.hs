module Course.Lectures.FunctionsIntro where

add :: (Int, Int) -> Int
add (x,y) = x+y

zeroto :: Int -> [Int]
zeroto n = [0..n]


add1 :: Int -> (Int->Int)
add1 x y = x + y

mult :: Int -> Int -> Int -> Int
mult x y z = x * y * z

add2 :: Int -> (Int->Int)
add2 x y = add (x, y)
