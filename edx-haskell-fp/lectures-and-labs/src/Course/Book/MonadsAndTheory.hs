module Course.Book.MonadsAndTheory where
import Data.Char

-- 1. FUNCTORS (From category theory)
-- Defining specific functions that later can be generalized to map
inc :: [Int] -> [Int]
inc [] = []
inc (n:ns) = n+1 : inc ns

sqr :: [Int] -> [Int]
sqr [] = []
sqr (n:ns) = n^2 : sqr ns

map1 :: (a->b)-> [a] -> [b]
map1 f [] = []
map1 f (x:xs) = f x : map1 f xs

inc1 = map (+1)
sqr1 = map (^2)

-- An example of a Functor is the defined as fmap which is more general than the map over lists
-- class Functor f where
  -- fmap :: (a->b)-> f a -> f b

-- Example:: Extending lists [] to the type functor
-- instance Functor [] where
    -- fmap :: (a->b)->[a]->[b]
  --fmap = map

-- Example with the Maybe type
-- It duplicates the code given in Prelude so we declare it as Maybe1
data Maybe1 a = Nothing1 | Just1 a

instance Functor Maybe1 where
  --fmap :: (a->b)-> Maybe1 a -> Maybe1 b
  fmap _ Nothing1 = Nothing1
  fmap g (Just1 x) = Just1 (g x)

-- Example binary Tree as functor
data Tree a = Leaf a | Node (Tree a) (Tree a) deriving Show

instance Functor Tree where
  --fmap :: (a->b)-> Tree a -> Tree b
  fmap g (Leaf x) = Leaf (g x)
  fmap g (Node l r) = Node (fmap g l) (fmap g r)

-- Making the IO type a functor
-- instance Functor IO where
   --fmap :: (a->b)-> IO a -> IO b
   --fmap g mx = do {x <- mx; return (g x)}

-- Generalizing the first examples to use fmap than the map functor
inc2 :: Functor f => f Int -> f Int
inc2 = fmap (+1)

sqr2 :: Functor f => f Int -> f Int
sqr2 = fmap (^2)

-- FUNCTOR LAWS
-- Does not suffice to provide a definition for a functor fmap, i.e., fmap must have the following properties:
-- Considering h::a->b and g::b->c
-- fmap id::a->a = id::f a->f a
-- fmap (g . h) = fmap g . fmap h

-- Later we'll learn how to formally prove that our definition of fmap is a indeed a Functor which means, it
-- does check all functor laws.

-- 2. APPLICATIVES
-- Suppose we want to generalize the idea of mapping a function over a structure that instead of having just
-- one argument, had multiple arguments. We would have to define:
-- fmap0 :: a -> f a   > A degenerate case when the function being mapped has no arguments
-- fmap1 :: (a->b) -> f a -> f b     > An alias for fmap
-- fmap2 :: (a->b->c) -> f a -> f b -> f c
-- fmap3 :: (a->b->c->d)-> f a -> f b -> f c -> f d
-- ...

-- One approach would be to define Functor0, Functor1, Functor2, etc... However, this would be unsatisfatory since we
-- would need to declare manually each version of the Functor class, which follow a similar pattern. Besides, it is not
-- clear a priori the number of classes needing declaration, even more, their number is infinite, so this is not viable.
-- Once again, fmap is a generalization of the built-in function application, thus we might expect that some form of
-- currying can be used to achieve the desired behaviour: having a mapping over multiple arguments.
-- It turns out that a version of fmap with any desired number of arguments can be constructed in termos of two basic
-- functions with the following types:
-- pure :: a -> f a
-- (<*>) :: f (a->b) -> f a -> f b
-- Therefore we can define fmapi by:
-- fmap0 = pure
-- fmap1 g x = pure g <*> x
-- fmap2 g x y = pure g <*> x <*> y
-- fmap3 g x y z = pure g <*> x <*> y <*> z
-- ...
-- Now we can define any fmapi as needed from pure and (<*>), and then define a special class of functors called the
-- Ãpplicative Functors or applicatives for short.
-- class Functor f => Applicative f where
  -- pure :: a -> f abs
  -- (<*>) :: f (a->b) -> f a -> f b
-- Note that fmap g is a special case of (pure g <*>)
-- Examples
-- Maybe
instance Applicative Maybe1 where
  -- pure :: a -> Maybe a
  pure = Just1

  -- (<*>) :: Maybe1 (a->b) -> Maybe1 a -> Maybe1 b
  Nothing1 <*> _ = Nothing1
  (Just1 g) <*> mx = fmap g mx

-- Lists
-- instance Applicative [] where
  ---- pure :: a -> [a]
  --pure x = [x]

  ---- (<*>) :: [a->b] -> [a] -> [b]
  --gs <*> xs = [g x | g <- gs, x <- xs]

-- Computes all possible ways of multiplying two lists of integers
prods :: [Int] -> [Int] -> [Int]
prods xs ys = [x * y | x <- xs, y <- ys]
-- Since Lists are applicative we can also give an applicative definition:
prods1 xs ys = pure (*) <*> xs <*> ys

-- IO example
-- instance Applicative IO where
  ---- pure :: a -> IO a
  -- pure = return

  ---- (<*>) :: IO (a->b) -> IO a -> IO b
  -- mg <*> mx = do {g <- mg; x <- mx; return (g x)}

-- Rapidly defining a function that reads a given number of characters from the keyboard:
getChars :: Int -> IO String
getChars 0 = return []
getChars n = pure (:) <*> getChar <*> getChars (n-1)

-- EFFECTFUL PROGRAMMING
-- Standard library:
--sequenceA :: Applicative f => [f a] -> f [a]
--sequenceA [] = pure []
--sequenceA (x:xs) = pure (:) <*> x <*> sequenceA xs

getChars1 :: Int -> IO String
getChars1 n = sequenceA (replicate n getChar)

-- Applicative laws
-- pure id <*> x = x
-- pure (g x) = pure g <*> pure x
-- x <*> pure y = pure (\g -> g y) <*> x
-- x <*> (y <*> z) = (pure (.) <*> x <*> y) <*> z


-- Additional law
-- fmap g x = pure g <*> x
-- TODO Look further over the previous laws

-- 3. MONADS

data Expr = Val1 Int | Div1 Expr Expr

eval :: Expr -> Int
eval (Val1 n) = n
eval (Div1 x y) = eval x `div` eval y

safediv :: Int -> Int -> Maybe Int
safediv _ 0 = Nothing
safediv n m = Just (n `div` m)

eval1 :: Expr -> Maybe Int
eval1 (Val1 n) = Just n
eval1 (Div1 x y) = case eval1 x of
                    Nothing -> Nothing
                    Just n -> case eval1 y of
                                Nothing -> Nothing
                                Just m -> safediv n m

-- To avoid being too verbose we redefine eval in applicative style, since Maybe is applicative
-- However Applicatives are used to apply pure functions to effectul arguments, contrary to this case
-- where safediv may fail
--eval :: Expr -> Maybe Int
--eval (Val1 n) = pure n
--eval (Div1 x y) = pure safediv <*> eval x <*> eval y

---- Abstracting out the pattern of mapping nothing to itself and Just x to some function of x we get the operator (>>=)
-- (>>=) :: Maybe a -> (a -> Maybe b) -> Maybe b
-- mx >>= f = case mx of
--                Nothing -> Nothing
--                Just x -> f x

-- (>>=) takes and argument that may fail and a function (a->b) that may fail, and return a result of type b that
-- may fail. We call it the bind operator. If the argument fails we propagate the failure, otherwise we apply the
-- function to the resulting value

-- Finally using the bind operator we redefine eval
eval2 :: Expr -> Maybe Int
eval2 (Val1 n) = Just n
eval2 (Div1 x y) = eval2 x >>= \n -> eval2 y >>= \m -> safediv n m

-- Using the equivalent do notation to redefine eval
eval3 :: Expr -> Maybe Int
eval3 (Val1 n) = Just n
eval3 (Div1 x y) = do n <- eval3 x
                      m <- eval3 y
                      safediv n m

-- More generally, the do notation is not specific to the types IO and Maybe, but can be used with any applicative type
-- that forms a Monad. In Haskell, this concept is captured by the following built-in declaration:
-- class Applicative m => Monad m where
--    return :: a -> m a
--    (>>=) :: m a -> (a -> m b) -> m b
--    return = pure

-- That is, a monad is an applicative type m that supports return and (>>=) functions of the specific types.
-- The default definition return = pure means that return is normally just another name for the applicative function pure,
-- but can be overriden in instances declarations if desired.

-- Examples
instance Monad Maybe1 where
  -- (>>=) :: Maybe a -> (a -> Maybe b) -> Maybe b
  Nothing1 >>= _ = Nothing1
  (Just1 x) >>= f = f x

--instance Monad [] where
  ----(>>=): [a] -> (a -> [b]) -> [b]
  --xs >>= f = [y | x <- xs, y <- f x]

pairs :: [a] -> [b] -> [(a,b)]
pairs xs ys = do x <- xs
                 y <- ys
                 return (x,y)

-- Similar definition to the comprehension notation, specific to the type of lists:
pairs1 :: [a]->[b]->[(a,b)]
pairs1 xs ys = [(x,y) | x <- xs, y <- ys]

-- 4. THE STATE MONAD
-- Consider the problem of writing functions that manipulate some form of state that can be changed over time.
-- For simplicity, we assume that the state is just an integer value, but this can be modified as required:
type State = Int
-- Most basic function  type on this type is a state transformer, abbreviated by ST:
newtype ST a = S (State -> (a, State))

app :: ST a -> State -> (a, State)
app (S st) = st

-- Towards making State into a Monad, we first define it as a functor
instance Functor ST where
  -- fmap :: (a->b)-> ST a -> ST b
  fmap g st = S (\s -> let (x, s') = app st s in (g x, s'))

-- Making it into an Applicative:
instance Applicative ST where
  -- pure :: a -> ST a
  pure x = S (\s -> (x, s))

  -- (<*>) :: ST (a->b)-> ST a -> ST b
  stf <*> stx = S (\s ->
    let (f, s') = app stf s
        (x, s'') = app stx s' in (f x, s''))


-- At last, its Monadic instance
instance Monad ST where
  -- (>>=) :: ST a -> (a -> ST b) -> ST b
  st >>= f = S (\s -> let (x, s') = app st s in app (f x) s')


-- Relabelling trees
-- Example of stateful programming:
data Tree1 a = Leaf1 a | Node1 (Tree1 a) (Tree1 a) deriving Show

tree :: Tree Char
tree = Node (Node (Leaf 'a') (Leaf 'b')) (Leaf 'c')

-- Consider a function that relabels each leaf in such a tree with a unique or fresh integer.
-- We implement this as a function that takes the next fresh integer as an additional argument, and returning the next
-- fresh integer as an additional result:
rlabel :: Tree a -> Int -> (Tree Int, Int)
rlabel (Leaf _) n = (Leaf n, n+1)
rlabel (Node l r) n = (Node l' r', n'')
                      where
                        (l', n') = rlabel l n
                        (r', n'') = rlabel r n'

-- Formulate the problem in terms of the ST type
fresh :: ST Int
fresh = S (\n -> (n, n+1))

-- Given that ST is an applicative functor, we can now define a new version of the relabelling function that is written
-- in applicative style
alabel :: Tree a -> ST (Tree Int)
alabel (Leaf _) = Leaf <$> fresh -- NOTE: g <$> x <-> pure g <*> x
alabel (Node l r) = Node <$> alabel l <*> alabel r

-- Monadic version of the labelling function
mlabel :: Tree a -> ST (Tree Int)
mlabel (Leaf _) = do n <- fresh
                     return (Leaf n)
mlabel (Node l r) = do l' <- mlabel l
                       r' <- mlabel r
                       return (Node l' r')

-- 5. Generic Functions
-- Many generic functions capable of be used with any Monad are provided in the Control.Monad library.
-- For instance, a monadic version of the map version on list can be defined as follows:
mapM1 :: Monad m => (a -> m b)->[a]-> m [b]
mapM1 f [] = return []
mapM1 f (x:xs) = do y <- f x
                    ys <- mapM1 f xs
                    return (y:ys)

-- Example:
conv :: Char -> Maybe Int
conv c
  | isDigit c = Just (digitToInt c)
  | otherwise = Nothing

const1 :: Maybe [Int]
const1 = mapM conv "1234"

-- Monadic version of the filter function
filterM :: Monad m => (a -> m Bool) -> [a] -> m [a]
filterM p [] = return []
filterM p (x:xs) = do b <- p x
                      ys <- filterM p xs
                      return (if b then x:ys else ys)

const1Filter:: [[Int]]
const1Filter = filterM (\x -> [True, False]) [1,2,3]

-- The prelude function concat :: [[a]] -> [a] on lists is generalised to an arbitrary monad as follows:
join :: Monad m => m (m a) -> m a
join mmx = do mx <- mmx
              x <- mx
              return x

-- 6. MONAD LAWS
-- Similarly to functor and applicatives, the two monadic primitives are required to satisfy some equational laws in
-- order to be valid definitions:
-- return x >>= f   = f x
-- mx >>= return    = mx
-- (mx >>= f) >>= g = mx >>= (\x -> (f x >>= g))
