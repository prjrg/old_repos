module Course.Lectures.FunctionsBasics where

-- Conditional expressions and guarded equations
signum :: Real a => a -> a
signum n = if n < 0 then -1 else if n == 0 then 0 else 1

abs :: Real a => a -> a
abs n | n >= 0 = n
      | otherwise = -n

-- Pattern Matching

not :: Bool -> Bool
not False = True
not True = False

(&&) :: Bool -> Bool -> Bool
True && b = b
False && _ = False

(||) :: Bool -> Bool -> Bool
False || b = b
True || _ = True


head :: [a] -> a
head (x:_) = x

tail :: [a] -> [a]
tail (_:xs) = xs

-- Lambda Expressions
add :: Num a => a -> a -> a
add = \x -> (\y -> x + y)


const :: a -> (b -> a)
const x = \_ -> x

odds n = map (\x -> x*2 + 1) [0..n-1]

-- Sections
a = 1 + 2

b = (+) 1 2

c = (1+) 2

d = (+2) 1

halve = (/2)
doubling = (*2)

