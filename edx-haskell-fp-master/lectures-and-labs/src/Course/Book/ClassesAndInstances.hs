module Course.Book.ClassesAndInstances where

-- Equivalent definition of class Eq from Prelude
class Eq0 a where
  (==!), (/=!) :: a -> a -> Bool
  x /=! y = not (x ==! y)

instance Eq0 Bool where
  False ==! False = True
  True ==! True = False
  _ ==! _ = False

class Eq0 a => Ord0 a where
  (<!), (<=!), (>!), (>=!) :: a -> a -> Bool
  min0, max0 :: a -> a -> a

  min0 x y | x <=! y = x
           | otherwise = y

  max0 x y | x <=! y = y
           | otherwise = x

instance Ord0 Bool where
  False <! True = True
  _ <! _ = False

  b <=! c = (b <! c) || (b ==! c)
  b >! c = b <! c
  b >=! c = c <=! b

data Bool0 = False0 | True0 deriving (Eq, Ord, Show, Read)

